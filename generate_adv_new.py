import sys, numpy as np, argparse, params, torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from funcs import *; from models import *
from torchvision.utils import save_image
import os



#CUDA_VISIBLE_DEVICES=1 python generate_adv_new.py --config configs/imagenet.json --attack_types l1 --path models/m_resnet-50/uar/linf --batch_size 100

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

imagenet_mu = torch.tensor(IMAGENET_MEAN).view(3,1,1)
imagenet_std = torch.tensor(IMAGENET_STD).view(3,1,1)

def ImagenetTransform(x):
    mu = imagenet_mu.to(x.device)
    std = imagenet_std.to(x.device)
    x = (x-mu)/std
    return x

def InverseImagenetTransform(x):
    mu = imagenet_mu.to(x.device)
    std = imagenet_std.to(x.device)
    x = x*std + mu
    return x


def generate_lp_imagenet(args, x_loc, y_loc, train, loader, model, criterion = nn.CrossEntropyLoss(), apgd = False):
    model.eval()
    num_samples = len(loader.dataset)
    print(num_samples)
    x_loc += "/train" if train else "/test"
    if not os.path.exists(x_loc):
        os.makedirs(x_loc)
    print (x_loc, y_loc)
    
    mapper_lp = {"linf":pgd_linf, "l1":pgd_l1, "l2": pgd_l2}
    attack = mapper_lp[args.attack_types[0]]

    i  = 0
    with tqdm(loader, unit="batch") as tepoch:
        for batch in tepoch:
            X,Y = batch[0].to(args.device), batch[1].to(args.device)
            delta = attack(model, X, Y, args)
            X_adv = X+delta
            true_acc = (model(X).max(1)[1] == Y).sum() / Y.shape[0]
            acc = (model(X_adv).max(1)[1] == Y).sum() / Y.shape[0]
            tepoch.set_postfix(clean_accuracy = 100*true_acc.item(), accuracy=100. *acc.item())
            # save batch as images
            X_adv = InverseImagenetTransform(X_adv)
            X_adv = X_adv.cpu()
            for im in X_adv:
                save_image(im, f'{x_loc}/IM_{i}.JPEG')
                # pil = transforms.ToPILImage()(im)
                # pil.save(f'{x_loc}/IMP_{i}.JPEG')
                i+=1
    
    print("Number of images saved =", i)

def generate_lp(args, x_loc, y_loc, train, loader, model, criterion = nn.CrossEntropyLoss(), apgd = False):
    if args.dataset.lower() == "imagenet": return generate_lp_imagenet(args, x_loc, y_loc, train, loader, model, criterion, apgd)
    model.eval()
    print (len(loader))
    print (x_loc, y_loc)
    num_samples = len(loader.dataset)
    print(num_samples)
    if train:
        x_loc += "_train.pt"
        y_label_loc = y_loc + "_train_label.pt"
        y_loc += "_train.pt"

    else:
        x_loc += "_test.pt"
        y_label_loc = y_loc + "_test_label.pt"
        y_loc += "_test.pt"
    
    y = torch.zeros(num_samples)
    y_label = torch.zeros(num_samples)
    if args.dataset == "CIFAR10":
        x = torch.zeros(num_samples,3,32,32)
    elif args.dataset == "MNIST": 
        x = torch.zeros(num_samples,1,28,28)
    elif args.dataset.lower() == "imagenet":
        x = torch.zeros(num_samples,3,224,224)
    
    elif args.dataset.lower() == "imagenette":
        x = torch.zeros(num_samples,3,128,128)

    # adversary_l2 = AutoAttack(model, norm='L2', eps=args.epsilon_l_2, version='standard', verbose = False)  
    # adversary_l2.attacks_to_run = ['apgd-ce'] 
    # adversary_linf = AutoAttack(model, norm='Linf', eps=args.epsilon_l_1, version='standard', verbose = False) 
    # adversary_linf.attacks_to_run = ['apgd-ce'] 
    # delta = adversary.run_standard_evaluation(X.clone(), Y, bs=args.batch_size) - X
    
    if args.attack_types[0][0] == "l":
        # mapper_lp = {"linf":pgd_linf, "l1":pgd_l1, "l2": pgd_l2}
        mapper_lp = {"linf":apgd_linf, "l1":apgd_l1, "l2": apgd_l2}
        print("apgd")
        attack = mapper_lp[args.attack_types[0]]
    else:
        from perceptual_advex.attacks import StAdvAttack, ReColorAdvAttack, JPEGLinfAttack
        attack_jpeg = JPEGLinfAttack(model,"cifar",bound=0.25)
        attack_stadv = StAdvAttack(model,bound=0.05)
        attack_recolor = ReColorAdvAttack(model,bound=0.06)
        mapper = {"jpeg":attack_jpeg, "stadv":attack_stadv, "recolor": attack_recolor}
        def attack(model,X,Y,args): return mapper[args.attack_types[0]](X,Y) - X

    distance_map = {"linf":0,"l1":1,"l2":2,"jpeg":3,"stadv":4,"recolor":5}
    y += distance_map[args.attack_types[0]]
    i  = 0
    for batch in tqdm(loader):
        X,Y = batch[0].to(args.device), batch[1].to(args.device)
        delta = attack(model, X, Y, args)
        X_adv = X+delta
        batch_size = X.shape[0]
        x[i*batch_size: (i+1)*batch_size] = X_adv.data.cpu()
        y_label[i*batch_size: (i+1)*batch_size] = Y.cpu()
        i+=1

        
    rand=torch.randperm(num_samples)
    x = x[rand]
    y = y[rand]
    y_label = y_label[rand]

    torch.save(x, x_loc)
    torch.save(y, y_loc)
    torch.save(y_label, y_label_loc)

def generate_pgd(args):
    train_loader, test_loader = get_dataloaders(args, no_transform = True)
    device = torch.device("cuda:{}".format(args.gpu_id) if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(int(args.gpu_id))
    args.device = device
    #args.attack_types[0] in l1,l2,linf,jpeg,recolor,stadv
    import os
    if args.dataset == "CIFAR10":
        root = f"../data/CIFAR10_ADVsmallstep_apgd/{args.attack_types[0]}"  
    elif args.dataset.lower() == "mnist":
        root = f"../data/MNIST_ADV/{args.attack_types[0]}"
    
    elif args.dataset.lower() == "imagenette":
        root = f"../data/IMAGENETTE_ADV/{args.attack_types[0]}"
    else:
        root = f"../data/IMAGENET_ADV_2/{args.attack_types[0]}"

    if(not os.path.exists(root)):
        os.makedirs(root)
    if args.mode.lower() == "pipeline":
        f_e = FeatureExtractor(args); f_e = nn.DataParallel(f_e).cuda()
        p_c = PerturbClassifier(args); p_c = nn.DataParallel(p_c).cuda()
        p_c.load_state_dict(torch.load(args.path + ".pt", map_location =args.device))
        model = Pipeline(args, f_e, p_c); model = nn.DataParallel(model).cuda()
        model.eval()
        # noise = "nn" if args.use_noise else ""
        x_loc = f"{root}/pipeline_x"
        y_loc = f"{root}/pipeline_y"
    elif args.mode == "rand":
        f_e = FeatureExtractor(args); f_e = nn.DataParallel(f_e).cuda()
        p_c_list = nn.ModuleList()
        for i in range(2):
            id = str(801 + i)
            loc = f"{args.path}model_{id}/model_info.txt"
            parser = params.parse_args()
            temp_args = parser.parse_args()
            temp_args = params.add_params_file(temp_args,loc)
            p_ci = PerturbClassifier(temp_args); p_ci = nn.DataParallel(p_ci).cuda()
            p_ci.load_state_dict(torch.load(args.path + f"model_{id}/final.pt", map_location =args.device))
            p_ci.eval()
            p_c_list.append(p_ci)
        model = RandPipeline(args, f_e, p_c_list); model = nn.DataParallel(model).cuda()
        model.eval()
        x_loc = f"{root}/randpipeline_x"
        y_loc = f"{root}/randpipeline_y"
    else:
        
        if args.dataset == "imagenette":
            weights=torch.load('/home/pratyus2/.fastai/data/imagenette2-160/models/imagenette_model.pth')
            model = xresnet34(n_out = 10)
            model.load_state_dict(weights['model'])
            model.cuda()
            base_name = "clean"
        else:
            model = get_model(args)
            model = nn.DataParallel(model).cuda()
            location = f"{args.path}.pt"
            if args.model_type!="resnet50":
                model.load_state_dict(torch.load(location))
            print (location)
            base_name = args.path.split("/")[-1].lower()

        model.eval()
        print(base_name)
        x_loc = f"{root}/{base_name}_x"
        y_loc = f"{root}/{base_name}_y"        

    generate_lp(args, x_loc, y_loc, True, train_loader,  model, apgd = False)
    generate_lp(args, x_loc, y_loc, False, test_loader,  model, apgd = False)


if __name__ == "__main__":
    parser = params.parse_args()
    args = parser.parse_args()
    args = params.add_config(args) if args.config_file != None else args
    # from fastai.vision.all import *
    
    generate_pgd(args)

# def restructure(name,mode): 
#     x_tr = torch.load(f"../MNIST_ADV2/{name}_x_{mode}.pt") 
#     y_tr = torch.load(f"../MNIST_ADV2/{name}_y_{mode}.pt") 
#     # y_l_tr = torch.load(f"../MNIST_ADV2/{name}_y_{mode}_label.pt")
#     name_map = {"triple":"avg","msd_v0":"msd","worst":"max","linf":"linf","l1":"l1","l2":"l2","vanilla":"vanilla"} 
#     name = name_map[name]
#     id_linf_tr =  (y_tr == 0)
#     id_l1_tr =  (y_tr == 1)
#     id_l2_tr =  (y_tr == 2)
#     ids = [id_linf_tr, id_l1_tr, id_l2_tr]
#     new_dirs = ["linf","l1","l2"]
#     for i,dir in enumerate(new_dirs):
#         torch.save(x_tr[ids[i]], f"{dir}/{name}_x_{mode}.pt")
#         torch.save(y_tr[ids[i]], f"{dir}/{name}_y_{mode}.pt")
#         # torch.save(y_l_tr[ids[0]], f"{dir}/{name}_y_{mode}_label.pt")


# def re_all():
#     restructure("vanilla","train")
#     restructure("vanilla","test")
#     restructure("msd_v0","train")
#     restructure("msd_v0","test")
#     restructure("triple","train")
#     restructure("triple","test")
#     restructure("worst","train")
#     restructure("worst","test")