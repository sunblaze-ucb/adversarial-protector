from __future__ import absolute_import
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from time import time
import argparse
from funcs import *
from models import *
import params


def trainer(params):
    print(params)
    lr_max = params.lr_max
    lr_min = params.lr_min
    epochs = params.epochs
    device = torch.device("cuda:{0}".format(params.gpu_id) if torch.cuda.is_available() else "cpu")
    print(device)
    torch.cuda.set_device(device)
    torch.manual_seed(params.seed)
  
    train_loader, test_loader = get_dataloaders(params)
    def myprint(a):
        print(a)
        file.write(a)
        file.write("\n")
        file.flush()

    def quick_eval(params, test_loader, model, train = False):
        tr_loader, te_loader = get_dataloaders(params)
        loader = te_loader if not train else tr_loader
        test = "Test" if not train else "Train"
        if params.mode == 'pipeline':
            linf_acc, l1_acc, l2_acc, l2ddn_acc, linf_aux_acc, l1_aux_acc, l2_aux_acc, l2ddn_aux_acc = \
                                    full_pipe_test(params, loader,  model, stop = True) 
            myprint(f'{test}  1: {l1_acc:.4f}, {test}  2: {l2_acc:.4f}, {test}  2 DDN: {l2ddn_acc:.4f}, {test}  inf: {linf_acc:.4f}')    
            myprint(f'Aux  1: {l1_aux_acc:.4f}, Aux  2: {l2_aux_acc:.4f}, Aux  2 DDN: {l2ddn_aux_acc:.4f}, Aux  inf: {linf_aux_acc:.4f}') 
        else:
            raise("Not implemented")  

        flag = min(linf_aux_acc, l1_aux_acc, l2_aux_acc) > 0.8
        return flag

    #### TRAIN CODE #####
    # attack = avg_pipeline
    root = f"models/m_{params.model_type}/Pipeline" if params.mode == 'pipeline' else f"models/m_{params.model_type}/Base"

    import glob, os, json
    num = params.model_id
    model_dir = f"{root}/model_{num}"

    if(not os.path.exists(model_dir)):
        os.makedirs(model_dir)
    file = open(f"{model_dir}/logs.txt", "a")    
    with open(f"{model_dir}/model_info.txt", "w") as f:
        json.dump(params.__dict__, f, indent=2)

    print(model_dir)

    params.device = device
    if params.lr_mode == 0:
        lr_schedule = lambda t: np.interp([t], [0, epochs*2//5, epochs], [lr_max, lr_max, lr_max/10])[0]
    elif params.lr_mode == 3:
        lr_schedule = lambda t: np.interp([t], [0, epochs//2, epochs], [lr_min, lr_max, lr_min])[0]
    elif params.lr_mode == 2:
        lr_schedule = lambda t: np.interp([t], [0, epochs*2//5, epochs*4//5, epochs], [lr_min, lr_max, lr_max/10, lr_min])[0]
    elif params.lr_mode == 1:
        lr_schedule = lambda t: np.interp([t], [0, epochs*1//3, epochs*1//3 + 0.0001, epochs*2//3, epochs*2//3 + 0.0001, epochs], [lr_max, lr_max, lr_max/10, lr_max/10, lr_max/100, lr_max/100])[0]

    t_start = 0
   
    
    
    model = get_model(params)
    model = nn.DataParallel(model).to(device)
    if params.opt_type == "SGD":
        opt = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    else:
        opt = optim.Adam(model.parameters(), lr=0.1)
    print ("Training Single Model")
    assert(params.mode=="base")
    attack_list = {"linf":pgd_linf, "l1":pgd_l1, "l2":pgd_l2, "vanilla":None}
    attack = attack_list[params.distance]     

    if params.model_type == "wrn-70-16":
        import robustbench as rb
        model = rb.utils.load_model(model_name='Rebuffi2021Fixing_70_16_cutmix_extra', dataset='cifar10', threat_model='Linf')
        model = nn.DataParallel(model).cuda()
    
    torch.save(model.state_dict(), f"{model_dir}/final_model.pt")        
    for t in range(t_start,epochs):  
        lr = lr_schedule(t)
        train_loss, train_acc = epoch_adversarial(params, train_loader, model, lr_schedule, t, attack = attack, opt = opt)
        test_loss, test_acc   = epoch_adversarial(params, test_loader, model)
        if params.distance == "vanilla":
            myprint(f'Epoch: {t}, Train Loss: {train_loss:.3f} Train Acc: {train_acc:.3f} Test Acc: {test_acc:.3f}, lr: {lr:.5f}')    
        else:
            params_test = params
            test_loss_1, test_acc_1 = epoch_adversarial(params_test, test_loader, model, attack = pgd_l1, stop = True)
            test_loss_2, test_acc_2 = epoch_adversarial(params_test, test_loader, model, attack = pgd_l2, stop = True)
            test_loss_inf, test_acc_inf = epoch_adversarial(params_test, test_loader, model, attack = pgd_linf, stop = True)
            myprint(f'Epoch: {t}, Train Loss: {train_loss:.3f} Train Acc: {train_acc:.3f} Test Acc: {test_acc:.3f}, Test Acc 1: {test_acc_1:.3f}, Test Acc 2: {test_acc_2:.3f}, Test Acc inf: {test_acc_inf:.3f}, lr: {lr:.5f}')    
        if params.dataset == "MNIST":
            torch.save(model.state_dict(), f"{model_dir}/iter_{t}.pt")
        elif (t+1)%5 == 0:
            torch.save(model.state_dict(), f"{model_dir}/iter_{t}.pt")


    torch.save(model.state_dict(), f"{model_dir}/final_model.pt")        

if __name__ == "__main__":
    print(sys.argv)
    parser = params.parse_args()
    args = parser.parse_args()
    args = params.add_config(args) if args.config_file != None else args
    trainer(args)

