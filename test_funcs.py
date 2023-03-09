import sys
import numpy as np
import torch
import torch.nn as nn
from time import time
from funcs import *
from models import *
sys.path.append("./auto-attack/")
from autoattack import AutoAttack
# from foolbox_attacks import test_foolbox



def get_auto_class(params, model):
    version = 'plus' if params.restarts > 1 else 'standard'
    version = "rand" if params.use_noise == 1 else version
    if params.distance == "l2":
        adversary = AutoAttack(model, norm='L2', eps=params.epsilon_l_2, version = version)  
        if params.auto_specific is not None:
            adversary.attacks_to_run = [params.auto_specific] 
    elif params.distance == "linf":
        adversary = AutoAttack(model, norm='Linf', eps=params.epsilon_l_inf, version = version)
        if params.auto_specific is not None:
            adversary.attacks_to_run = [params.auto_specific] 
    else:
        adversary = AutoAttack(model, norm='L1', eps=params.epsilon_l_1, version = version) 
        adversary.attacks_to_run = ['fab-t'] 
    
    if 'fab' in adversary.attacks_to_run: adversary.attacks_to_run.remove('fab') #fab-t was better than fab always in my exps

    return adversary

def test_auto_attack(params, model, location, max_check = 1000, subset = -1, save_images = False, save_acc = True):
    model.eval()
    assert(params.distance in ["linf","l2","l1"])
    
    adversary = get_auto_class(params, model)

    _, test_loader = get_dataloaders(params)  
    loader = test_loader
    
    adv_images = []
    y_images = []
    num_images = 0
    misclassification_store = {}

    for images,labels in loader:     
         
        images = images.to(params.device); labels = labels.to(params.device)
        num_images += images.shape[0]
        if save_images:
            x_adv = adversary.run_standard_evaluation_individual(images.clone(), labels, bs=params.batch_size)
            x_adv = x_adv['fab-t']
            adv_images.append(x_adv.cpu().detach())
            y_images.append(labels.cpu().detach())
        else:
            adversary = get_auto_class(params, model)
            rets = adversary.run_standard_evaluation_individual(images.clone(), labels, bs=params.batch_size)
            torch.cuda.empty_cache() 
            for key in rets.keys():
                if key not in misclassification_store: misclassification_store[key] = []
                x_adv = rets[key]
                if params.mode.lower() != "base":
                    model.module.args.pool = 'max'
                    yp_aux, yp = model(x_adv, return_both = True)
                    model.module.args.pool = params.pool
                else:
                    yp = model(x_adv)
                    yp_aux = yp
                preds = yp.max(1)[1]; classes = yp_aux.max(1)[1]

                print(key, "Model Accuracy: ", (preds == labels).sum().item()*100/float(labels.numel()), "P_C class = 1: ", (classes == 1).sum().item()*100/float(labels.numel()))
                misclassification = (yp.max(1)[1] != labels).cpu().detach().unsqueeze(-1).numpy()
                misclassification_store[key].append(misclassification) 
                del yp, x_adv, yp_aux, preds, classes
            del images, labels
            torch.cuda.empty_cache() 

        
        if num_images >= max_check:
            break
    
    for key in rets.keys():
        attack_name = key
        print(f"Saving {key}")
        np.save(location + f"/{params.pool}_{params.distance}" + attack_name + ".npy", np.vstack(misclassification_store[key]))
            
    if save_images:
        (x_loc, y_loc) = (f"../data/CIFAR10_ADV_2/fab_t_{params.distance}_x", f"../data/CIFAR10_ADV_2/fab_t_{params.distance}_y")
        x_loc += "_train.pt"
        y_label_loc = y_loc + "_train_label.pt"
        y_loc += "_train.pt"
        
        x = torch.cat(adv_images)
        y = torch.zeros(x.shape[0]) if params.distance == "linf" else torch.ones(x.shape[0])*2
        y_label = torch.cat(y_images)

        torch.save(x, x_loc); torch.save(y, y_loc); torch.save(y_label, y_label_loc)

def clean_acc(args, model, location,loader = None):
    _, test_loader = get_dataloaders(args)
    loader = test_loader
    model.eval()
    test_acc, test_n = 0,0
    with torch.no_grad():
        for batch in loader:
            X,y = batch[0].to(args.device), batch[1].to(args.device)
            yp = model(X)
            test_acc += (yp.max(1)[1] == y).sum().item() 
            test_n += y.size(0)
        
    print (f"Clean Accuracy =  {test_acc / test_n}")


def quick_eval(args, model, location,loader = None):
    if loader == None:
        train_loader, test_loader = get_dataloaders(args)
        loader = test_loader
    la,ld=full_pipe_test(args, loader,  model, stop = True, return_misclassification = True)
    num_attacks = len(args.attack_types)

    for i in range(num_attacks):
        print(f'Test  {args.attack_types[i]}: {la[i]*100:.4f}')    
        if args.mode != "base":
            print(f'Aux  {args.attack_types[i]}: {la[i+num_attacks]*100:.4f}') 
        np.save(f"{location}/{args.pool}_{args.attack_types[i]}.npy", ld[i])


def quick_eval_dual(params, f_e, p_c, location, loader = None, lamb = 1):
    if loader == None:
        _, test_loader = get_dataloaders(params)
        loader = test_loader
    la,ld=full_pipe_dual_test(params, loader,  f_e, p_c, stop = True, return_perturbation = False)
    (linf_acc, l1_acc, l2_acc, linf_aux_acc, l1_aux_acc, l2_aux_acc) = la
    (linf_misclassification, l1_misclassification, l2_misclassification) = ld
    print(f'Test  1: {l1_acc:.4f}, Test  2: {l2_acc:.4f}, Test  inf: {linf_acc:.4f}')    
    if params.mode != "base":
        print(f'Aux  1: {l1_aux_acc:.4f}, Aux  2: {l2_aux_acc:.4f},  Aux  inf: {linf_aux_acc:.4f}') 

    np.save(location + "/" + "L1" + ".npy", l1_misclassification) 
    np.save(location + "/" + "L2" + ".npy", l2_misclassification) 
    np.save(location + "/" + "LINF" + ".npy", linf_misclassification) 
 

def analyze_image_perturbation(params, model):
    _, test_loader = get_dataloaders(params)

    la,ld=full_pipe_test(params, test_loader,  model, stop = True, return_advs = True)
    (linf_acc, l1_acc, l2_acc, l2ddn_acc, linf_aux_acc, l1_aux_acc, l2_aux_acc, l2ddn_aux_acc) = la
    (linf_advs, l1_advs, l2_advs, l2ddn_advs) = ld        # time_elapsed = time.time()-start   
    print(f'Test  1: {l1_acc:.4f}, Test  2: {l2_acc:.4f}, Test  2 DDN: {l2ddn_acc:.4f}, Test  inf: {linf_acc:.4f}')    
    print(f'Aux  1: {l1_aux_acc:.4f}, Aux  2: {l2_aux_acc:.4f}, Aux  2 DDN: {l2ddn_aux_acc:.4f}, Aux  inf: {linf_aux_acc:.4f}') 
    

    print(linf_advs.mean(), l1_advs.mean(), l2_advs.mean(), l2ddn_advs.mean())
    print(linf_advs.var(), l1_advs.var(), l2_advs.var(), l2ddn_advs.var())
 

def test_pgd_saver(model, location):
    #Saves the minimum epsilon value for successfully attacking each image via PGD based attack as an npy file 
    #in the folder corresponding to location
    device = params.device
    model = model.eval()
    eps_1 = [3,6,(10),12,20,30,50,60,70,80,90,100]
    eps_2 = [0.1,0.2,0.3,0.5,1.0,1.5,2.0,2.5,3,5,7,10]
    eps_3 = [0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    num_1 = [50,50,100,100,100,200,200,200,300,300,300,300]
    num_2 = [30,40,50,50,100,100,150,150,150,150,300,300]
    num_3 = [30,40,50,50,100,100,150,150,150,150,300,300]
    attacks_l1 = torch.ones((batch_size, 12))*1000
    attacks_l2 = torch.ones((batch_size, 12))*1000
    attacks_linf = torch.ones((batch_size, 12))*1000
    
    for index in range(len(eps_1)):
            _,test_loader = get_dataloaders(params)
            e_1 = eps_1[index]
            n_1 = num_1[index]
            eps, total_acc_1 = epoch_adversarial_saver(batch_size, test_loader, model, pgd_l1_topk, e_1, n_1, device = device, restarts = res)
            attacks_l1[:,index] = eps
    attacks_l1 = torch.min(attacks_l1,dim = 1)[0]
    np.save(location + "/" + "CPGDL1" + ".npy" ,attacks_l1.numpy())

    for index in range(len(eps_2)):        
            _,test_loader = get_dataloaders(params)
            e_2 = eps_2[index]
            n_2 = num_2[index]
            eps, total_acc_2 = epoch_adversarial_saver(batch_size, test_loader, model, pgd_l2, e_2, n_2, device = device, restarts = res)
            attacks_l2[:,index] = eps
    attacks_l2 = torch.min(attacks_l2,dim = 1)[0]
    np.save(location + "/" + "CPGDL2" + ".npy" ,attacks_l2.numpy())

    for index in range(len(eps_3)):
            _,test_loader = get_dataloaders(params)
            e_3 = eps_3[index]
            n_3 = num_3[index]
            eps, total_acc_3 = epoch_adversarial_saver(batch_size, test_loader, model, pgd_linf, e_3, n_3, device = device, restarts = res)
            attacks_linf[:,index] = eps
    attacks_linf = torch.min(attacks_linf,dim = 1)[0]
    np.save(location + "/" + "CPGDLINF" + ".npy" ,attacks_linf.numpy())

