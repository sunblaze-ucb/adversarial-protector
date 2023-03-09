import random, copy, sys, numpy as np
import torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as F
from tqdm import tqdm
from time import time
from attacks import *; from models import *
sys.path.append("./auto-attack/")
from autoattack import AutoAttack
from dataloader import get_dataloaders

def attacks_to_label_mapper(params):
    if params.club_l1_l2 == 0:
        aux_label = {pgd_linf:0, apgd_linf:0, square_linf:0, 
                    pgd_l1:1, apgd_l1:1, square_l1:1, 
                    pgd_l2:2, apgd_l2:2, square_l2:2,
                    ddn:2, jpeg:3, stadv:4, recolor:5} 
    else:
        aux_label =  {pgd_linf:0, apgd_linf:0, square_linf:0, 
                    pgd_l1:1, apgd_l1:1, square_l1:1,
                    pgd_l2:1, apgd_l2:1, square_l2:1,
                    ddn:1, jpeg:2, stadv:3, recolor:4}

    if params.num_base == 4:
        aux_label = {pgd_linf:0, pgd_l2:1, ddn:1, stadv:2, recolor:3}
    return aux_label

def epoch_adversarial(params, loader, model, lr_schedule = None, epoch_i = None, attack = None, opt=None, stop = False):
    """Adversarial training/evaluation epoch over the dataset"""
    train_loss = 0
    train_acc = 0
    train_n = 0
    i = 0
    func = tqdm if stop == False else lambda x:x
    for batch in func(loader):
        X,y = batch[0].to(params.device), batch[1].to(params.device)
        
        delta = attack(model, X, y, params) if attack is not None else 0
        yp = model(X+delta)
        
        loss = nn.CrossEntropyLoss()(yp,y)

        if opt:
            lr = lr_schedule(epoch_i + (i+1)/len(loader))
            opt.param_groups[0].update(lr=lr)
            opt.zero_grad()
            loss.backward()
            opt.step()

        train_loss += loss.item()*y.size(0)
        train_acc += (yp.max(1)[1] == y).sum().item()
        train_n += y.size(0)
        i += 1
        if stop:
            break
        
    return train_loss / train_n, train_acc / train_n

def epoch_adversarial_saver(batch_size, loader, model, attack, epsilon, num_iter, device = "cuda:0", restarts = 10):
    criterion = nn.CrossEntropyLoss()
    train_loss = 0
    train_acc = 0
    train_n = 0

    for i,batch in enumerate(loader): 
        X,y = batch[0].to(device), batch[1].to(device)
        delta = attack(model, X, y, epsilon = epsilon, num_iter = num_iter, device = device, restarts = restarts)
        output = model(X+delta)
        loss = criterion(output, y)
        train_loss += loss.item()*y.size(0)
        train_acc += (output.max(1)[1] == y).sum().item()
        correct = (output.max(1)[1] == y).float()
        eps = (correct*1000 + epsilon - 0.000001).float()
        train_n += y.size(0)
        break
    return eps,  train_acc / train_n

def full_pipe_dual_test(params_original, loader,  f_e, p_c, stop = True, return_misclassification = False, return_advs = False):
    p_c.eval()
    f_e.eval()
    model = [f_e, p_c]
    delta_l_1_store, delta_l_2_store, delta_l_inf_store = [],[],[]
    delta_l_p_map = {pgd_linf:delta_l_inf_store, pgd_l1:delta_l_1_store, pgd_l2:delta_l_2_store}
    params = copy.deepcopy(params_original)
    params.restarts = params.restarts + 1 
    test_n = 0

    attacks_list = [pgd_linf, pgd_l1, pgd_l2]
    aux_label = {pgd_linf:0, pgd_l1:1, pgd_l2:2} if params.club_l1_l2 == 0 else {pgd_linf:0, pgd_l1:1, pgd_l2:1}

    for i, batch in enumerate(loader):
        X,y = batch[0].to(params.device), batch[1].to(params.device)
        
        accuracies = [0, 0, 0] #Accuracy for Label Classification
        aux_accuracies = [0, 0, 0] #Accuracy for attack classification
        for i, attack in enumerate(attacks_list):#Change
            #L_p
            y_p = aux_label[attack] #Attack Type 

            delta_l_p = attack(model, X, y, params)#Change
            yp_l_p, yp_l_p_aux = get_dual_preds(f_e, p_c, X+delta_l_p, params.num_attacks, both = True)
            accuracies[i] += (yp_l_p.max(1)[1] == y).sum().item()
            aux_accuracies[i] += (yp_l_p_aux.max(1)[1] == y_p).sum().item() if params.mode == 'pipeline' else 0
            
            if return_misclassification:
                delta_l_p_map[attack].append((yp_l_p.max(1)[1] != y).cpu().detach().numpy()) 

            if return_advs:
                delta_l_p_map[attack].append((X + delta_l_p).cpu().detach().numpy()) 

            del yp_l_p_aux, delta_l_p, yp_l_p
            torch.cuda.empty_cache()
        
        test_n += y.size(0)
        if stop:
            break

    torch.cuda.empty_cache()
    l1 = list(np.array(accuracies + aux_accuracies)/float(test_n))
    
    if not (return_misclassification or return_advs) :
        return l1
    l2 = (np.vstack(delta_l_inf_store), np.vstack(delta_l_1_store), np.vstack(delta_l_2_store)) 
    return l1,l2

def full_pipe_test(params_original, loader,  model, stop = True, return_misclassification = False, return_advs = False):
    model.eval()
    # return_advs = 1
    # return_misclassification = 0
    params = copy.deepcopy(params_original)
    params.restarts = params.restarts + 1 #Change
    test_n = 0
    delta_store = [[] for i in range(len(params.attack_types))]
    accuracies = [0 for i in range(len(params.attack_types))] #Accuracy for Label Classification
    aux_accuracies = [0 for i in range(len(params.attack_types))] #Accuracy for attack classification
    attack_mapper = {"linf":pgd_linf,"l1":pgd_l1,"l2":pgd_l2,"jpeg":jpeg,"stadv":stadv,"recolor":recolor,"ddn":ddn}
    attacks_list = [attack_mapper[at] for at in params.attack_types]
    aux_label = attacks_to_label_mapper(params)

    for j, batch in enumerate(loader):
        X,y = batch[0].to(params.device), batch[1].to(params.device)
        for i, attack in enumerate(attacks_list):#Change
            start = time()
            y_p = aux_label[attack] #Attack Type 
            if params.mode in ["pipeline","rand"] and params.pool != "uniform": model.module.args.pool = params.pool
            delta_l_p = attack(model, X, y, params)#Change
            if params.mode in ["pipeline","rand"] and params.pool != "uniform": model.module.args.pool = "max"
            yp_l_p = model(X+delta_l_p)
            accuracies[i] += (yp_l_p.max(1)[1] == y).sum().item()
            yp_l_p_aux = model(X+delta_l_p, forward_classifier = True) if params.mode != 'base' else 0
            aux_accuracies[i] += (yp_l_p_aux.max(1)[1] == y_p).sum().item() if params.mode != 'base' else 0
            if params.perturb_stats:
                l1_n = norms_l1(delta_l_p).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                l2_n = norms_l2(delta_l_p).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                linf_n = norms_linf(delta_l_p).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            
            if return_misclassification:
                delta_store[i].append((yp_l_p.max(1)[1] != y).cpu().detach().unsqueeze(-1).numpy()) 

            if return_advs:
                delta_store[i].append((X + delta_l_p).cpu().detach().numpy()) 

            del yp_l_p_aux, delta_l_p, yp_l_p
            torch.cuda.empty_cache()
        print(accuracies, aux_accuracies, time() - start)
        test_n += y.size(0)
        if stop and test_n>=1000:
            break

    model.zero_grad()
    torch.cuda.empty_cache()
    if params.mode == "pipeline": model.module.args.pool = params.pool
    l1 = list(np.array(accuracies + aux_accuracies)/float(test_n))
    
    if not (return_misclassification or return_advs) :
        return l1
    l2 = [np.vstack(delta_store[i]) for i in range(len(params.attack_types))]
    return l1,l2


