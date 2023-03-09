import foolbox as fb
import foolbox.attacks as fa
version = int(fb.__version__.split(".")[0])
import numpy as np
import torch
import torch.nn as nn
from time import time


def get_attack(attack, fmodel):
    args = []
    kwargs = {}
    # L0
    if attack == 'SAPA':
        A = fa.SaltAndPepperNoiseAttack()
    
    # L2
    elif attack == 'IGD':
        A = fa.L2BasicIterativeAttack()
    elif attack == 'AGNA':
        A = fa.L2AdditiveGaussianNoiseAttack()
    elif attack == 'BA':
        A = fa.BoundaryAttack()
    elif 'DeepFool' in attack:
        A = fa.L2DeepFoolAttack()
    elif attack == "CWL2":
        A = fa.L2CarliniWagnerAttack()

    # L inf
    elif 'FGSM' in attack and not 'IFGSM' in attack:
        A = fa.FGSM()
    elif 'PGD' in attack:
        A = fa.LinfPGD()
    
    else:
        #The following attacks are no longer supported by foolbox v3
        # Run the following command on terminal: pip install foolbox==2
        assert(version <=2)
        if attack == 'PA':
            A = fa.PointwiseAttack(fmodel, distance = fb.distances.L0)
        elif attack == 'PAL2':
            A = fa.PointwiseAttack(fmodel, distance = fb.distances.MSE)
        elif 'IGM' in attack:
            A = fa.MomentumIterativeAttack(fmodel, distance = fb.distances.Linf)
        else:
            raise Exception('Not implemented')

    return A, 0,0,0

def parse_subset(subset):
    if subset == 0:
        attacks_list = ['PA','IGM']#, 'PAL2']
        types_list   = [ 1    , 2]# , 2]
    elif subset == 1:
        types_list   = [ 2 ]
        attacks_list = ['BA']
    elif subset == 2:
        attacks_list = ['IGD','AGNA','DeepFool','SAPA']
        types_list = [2,2,2,1]
    elif subset == 3 :
        attacks_list =['PGD','FGSM','CWL2']
        types_list = [0,0,2]
    elif subset == 4:
        types_list   = [ 2 ]
        attacks_list = ['CWL2']
    else:
        attacks_list = ['SAPA','IGD','AGNA','BA','DeepFool','CWL2','FGSM','PGD']
        types_list   = [ 1    , 2   , 2    ,  2  ,  2   , 2,   0      , 0  ]
    return attacks_list, types_list    

def test_foolbox(params, model, location, max_check, subset = -1):
    # Saves the misclassification corresponding to different attacks in foolbox
    # No Restarts in case of BA
    print(max_check, location)
    torch.manual_seed(0)
    batch_size = params.batch_size; device = params.device
    model = model.eval()
    preprocessing = dict()
    bounds = (0, 1)
    if version  <= 2:
        fmodel = fb.models.PyTorchModel(model,bounds=(0., 1.), num_classes=10, device=device)
    else:
        fmodel = fb.PyTorchModel(model, bounds=bounds, preprocessing=preprocessing, device = device)
        fmodel = fmodel.transform_bounds((0, 1))
        assert fmodel.bounds == (0, 1)

    attacks_list, types_list = parse_subset(subset)
    norm_dict = {0:norms_linf_squeezed, 1:norms_l1_squeezed, 2:norms_l2_squeezed}
    norm_dict_un = {0:norms_linf, 1:norms_l1, 2:norms_l2}
    epsilons_dict = {0:params.epsilon_l_inf, 1:params.epsilon_l_1, 2:params.epsilon_l_2} 

    for i in range(len(attacks_list)):
        attack_name = attacks_list[i]
        restarts = 1 if attack_name in ["BA","CWL2"] else params.restarts
        restarts = 2 if attack_name in ["PA"] else restarts
        print (attack_name)
        types = types_list[i]
        norm = norm_dict[types]; norm_un = norm_dict_un[types]
        _, test_loader = get_dataloaders(params.dataset, batch_size)

        output = np.ones((max_check))
        
        attack, metric, args, kwargs = get_attack(attack_name, fmodel)
        misclassification_store = []
        total, l_acc, l_aux_acc = 0,0,0
        for images,labels in test_loader:        
            images = images.to(device); labels = labels.to(device)
            start = time()
            distance = 1000*torch.ones(batch_size)
            misclassification = (labels != labels).cpu().detach().numpy()

            best_delta = torch.zeros_like(images)
            for r in range (restarts):
                if version <= 2:
                    advs = attack(images.cpu().numpy(), labels=labels.cpu().numpy())
                    advs = torch.from_numpy(advs).to(device)
                else:
                    criterion = fb.criteria.Misclassification(labels)
                    advs, clipped, is_adv = attack(fmodel, images, criterion, epsilons=epsilons_dict[types])
                delta = advs-images
                new_distance = norm(delta).cpu()
                best_delta[distance > new_distance] = delta[distance>new_distance]
                distance[distance > new_distance] = new_distance[distance>new_distance]

            delta = best_delta.clone()
            if types == 0:
                delta.data = delta.data.clamp(-epsilons_dict[types],epsilons_dict[types])
            else:
                delta.data *=  epsilons_dict[types] / norm_un(delta.detach()).clamp(min=epsilons_dict[types]) 
            
            if params.mode != 'base':
                model.module.args.pool = 'max'
                yp_aux, yp = model(images + delta, return_both = True)
                model.module.args.pool = params.pool
            else:
                yp = model(images + delta)
                yp_aux = yp
            l_acc += (yp.max(1)[1] == labels).sum().item()
            l_aux_acc += (yp_aux.max(1)[1] == types).sum().item()
            misclassification = (yp.max(1)[1] != labels).cpu().detach().unsqueeze(-1).numpy()
            misclassification_store.append(misclassification) 
            
            total += batch_size
            print(f"Num = {total} | Attack  = {attack_name} | model id   = {location} | mean distance = {distance.mean()}")
            print(f"Acc = {l_acc} | Aux Acc = {l_aux_acc}   |  Time taken = {time() - start}")
            
            if (total >= max_check):
                np.save(location + f"/{params.pool}_" + attack_name + ".npy", np.vstack(misclassification_store))
                break
        
        print(f'Attack Name: {attack_name} | Test  Accuracy: {l_acc:.4f} | Classification  Accuracy: {l_aux_acc:.4f}')
  
