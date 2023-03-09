from __future__ import absolute_import
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import argparse
from funcs import *
from models import *
import params
from test_funcs import quick_eval
from staged_trainer import get_augmented_loader


def trainer(args):
    lr_max = args.lr_max
    lr_min = args.lr_min
    epochs = args.epochs
    device = torch.device('cuda')
    torch.manual_seed(args.seed)

    #return_dataset is used so that we can augment more adversarial images in the previous dataset.
    train_loader, test_loader, d_train, d_test = get_dataloaders(args,adversarial = True, return_datasets = True)

    #print function to log the training process
    def myprint(a):
        print(a)
        file.write(a)
        file.write("\n")
        file.flush()

    #### TRAIN CODE #####
    import glob, os, json
    
    #logging folder. Staged uses a dynamic training set (augmented each epoch), while Static uses a fixed dataset.
    training_type = "Staged" if args.staged_train else "Static"
    model_dir = f"models/m_{args.model_type}/{training_type}/model_{args.model_id}"

    if(not os.path.exists(model_dir)):
        os.makedirs(model_dir)
    file = open(f"{model_dir}/logs.txt", "w")    
    
    with open(f"{model_dir}/model_info.txt", "w") as f:
        json.dump(args.__dict__, f, indent=2)
        print(args)
    
    args.device = device

    
    #Various Learning Rate 
    if args.lr_mode == 0:
        lr_schedule = lambda t: np.interp([t], [0, epochs*2//5, epochs], [lr_max, lr_max, lr_min])[0]
    elif args.lr_mode == 1:
        lr_schedule = lambda t: np.interp([t], [0, epochs*2//5, epochs*4//5, epochs], [lr_min, lr_max, lr_max/10, lr_min])[0]
        # lr_schedule = lambda t: np.interp([t], [0, epochs*2//5, epochs], [lr_min, lr_max, lr_min])[0]
    elif args.lr_mode == 2:
        # lr_schedule = lambda t: np.interp([t+1], [0, epochs*2//5, epochs*4//5, epochs], [lr_min, lr_max, lr_max/10, lr_min])[0]
        lr_schedule = lambda t: np.interp([t+1], [0, epochs*2//5, epochs], [lr_min, lr_max, lr_min])[0]
    elif args.lr_mode == 3:
        lr_schedule = lambda t: np.interp([t+1], [0, epochs], [lr_max, lr_max])[0]


    # In some experiments we will use the perturbation classifier on internal features of second level models. Otherwise directly on the input
    feature_extractor = FeatureExtractor(args) if args.features else None
    feature_extractor = nn.DataParallel(feature_extractor).cuda() if args.features else None

    p_c = get_perturb_classifier(args)
    
    if args.staged_train:
        #if we are using staged training we need the entire model to craft new attacks
        pipeline_model = Pipeline(args, feature_extractor, p_c)
        pipeline_model = nn.DataParallel(pipeline_model).cuda()
        model = pipeline_model.module.p_c
        # model = nn.DataParallel(model).cuda()
        model = model.cuda()
    else:
        #otherwise the perturbation classification module is suffficient
        model = nn.DataParallel(p_c).cuda()


    #Optimizer
    if args.opt_type == "SGD":
        opt = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    else:
        opt = optim.Adam(model.parameters(), lr=0.1, weight_decay=5e-4)
    
    
    t_start = 0
    dataset_iter = 0

    if args.resume:
        location = f"{model_dir}/iter_{str(args.resume_iter)}.pt"
        t_start = args.resume_iter + 1
        model.load_state_dict(torch.load(location, map_location = device))

    for epoch_i in range(t_start,epochs):  
        start_time = time()
        lr = lr_schedule(epoch_i + (epoch_i+1)/len(train_loader))

        train_loss, train_acc = epoch_adversarial_dual(args, train_loader, model, feature_extractor, lr_schedule, epoch_i, opt = opt)
        test_loss, test_acc   = epoch_adversarial_dual_test(args, test_loader, model, feature_extractor)
        
        model.zero_grad()
        myprint(f'Epoch: {epoch_i}, Train Loss: {train_loss:.3f} Train Acc: {train_acc:.3f} Test Acc1: {test_acc:.3f}  Time: {(time()-start_time):.3f}, lr: {lr:.6f}')    

        if train_acc > 0.9 and args.staged_train and (epoch_i < epochs - 1):
            train_loader, d_train = get_augmented_loader(args, pipeline_model, model_dir, dataset_iter, d_train)
            test_loader, d_test = get_augmented_loader(args, pipeline_model, model_dir, dataset_iter, d_test, test = True)
            
            _, train_acc   = epoch_adversarial_dual_test(args, train_loader, model, feature_extractor)
            _, test_acc   = epoch_adversarial_dual_test(args, test_loader, model, feature_extractor)
            
            myprint(f'Initial Train Acc = {train_acc} | Test Acc = {test_acc}')
            dataset_iter += 1

        # if ((epoch_i+1)%5)== 0:
        torch.save(model.state_dict(), f"{model_dir}/iter_{epoch_i}.pt")
    
    torch.save(model.state_dict(), f"{model_dir}/final.pt")
    


def epoch_adversarial_dual_test(args, loader, model, feature_extractor, stop = False):
    """Adversarial training/evaluation epoch over the dataset"""
    model.eval()
    test_loss, test_acc, test_n, i = 0,0,0,0
    criterion = args.criterion

    func = tqdm if stop == False else lambda x:x
    with torch.no_grad():
        for batch in func(loader):
            X,y = batch[0].to(args.device), batch[1].to(args.device)
            features = feature_extractor(X) if args.features else X
            yp = model(features)
            
            loss = nn.CrossEntropyLoss()(yp,y)

            test_loss += loss.item()*y.size(0)
            test_acc += (yp.max(1)[1] == y).sum().item() 
            test_n += y.size(0)
            i+=1
            if stop:
                break
        
    return test_loss / test_n, test_acc / test_n


def epoch_adversarial_dual(args, loader, model, feature_extractor, lr_schedule = None, epoch_i = None, opt=None, stop = False):
    """Adversarial training/evaluation epoch over the dataset"""
    model.train()
    train_loss, train_acc, train_n, i = 0,0,0,0
    criterion = args.criterion
    assert(opt)
    func = tqdm if stop == False else lambda x:x
    with tqdm(loader, unit="batch") as tepoch:
        for batch in tepoch:
            tepoch.set_description(f"Epoch {epoch_i}")
            X,y = batch[0].cuda(), batch[1].cuda()
            noise = get_noise_like(args,X) if args.use_noise else 0
            X = (X+noise).clamp(0,1)
            features = feature_extractor(X) if args.features else X
            if args.fft ==2:
                yp1 = model(features, forward_two = False); loss1 = nn.CrossEntropyLoss()(yp1,y)
                yp2 = model(features, forward_one = False); loss2 = nn.CrossEntropyLoss()(yp2,y)
                loss = loss1 + loss2
                yp = yp1 + yp2
            else:
                yp = model(features)
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
            i+=1
            if args.fft == 2:
                tepoch.set_postfix(loss1=loss1.item(), loss2=loss2.item(), accuracy1=100. * (yp1.max(1)[1] == y).sum().item()/X.shape[0], accuracy2=100. * (yp2.max(1)[1] == y).sum().item()/X.shape[0])
            else:
                tepoch.set_postfix(loss=loss.item(), accuracy=100. * (yp.max(1)[1] == y).sum().item()/X.shape[0])
            if stop:
                break

    return train_loss / train_n, train_acc / train_n


if __name__ == "__main__":
    parser = params.parse_args()
    args = parser.parse_args()
    args = params.add_config(args) if args.config_file != None else args
    
    trainer(args)