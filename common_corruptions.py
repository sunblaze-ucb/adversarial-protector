import sys, os, argparse, params, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from funcs import *
from models import *
from test_funcs import *

def get_p_labels(args):
    batch_size = args.batch_size
    
    if args.dataset == "CIFAR10":
        c_p_dir =  'CIFAR-10-C/' 
        p_list = ['fog','jpeg_compression','zoom_blur','speckle_noise','glass_blur','spatter',
        'shot_noise','defocus_blur','elastic_transform','gaussian_blur','frost',
        'saturate','brightness','snow','gaussian_noise','motion_blur','contrast',
        'impulse_noise','pixelate']

    else :
        c_p_dir = 'mnist_c/'
        p_list = ['brightness','canny_edges','dotted_line','fog','glass_blur','identity',
        'impulse_noise','motion_blur','rotate','scale','shear','shot_noise','spatter','stripe','translate','zigzag']
        c_p_dir = '../data/' + c_p_dir
        # for p in p_list:
        #     t = np.load(c_p_dir+p+"/test_images.npy")
        #     np.save(c_p_dir+p+".npy",t)

    c_p_dir = '../data/' + c_p_dir
    _, test_loader = get_dataloaders(args)    
    labels = torch.from_numpy(np.float32(np.load(os.path.join(c_p_dir + 'labels.npy'))))
    num_levels = 5 if args.dataset=='CIFAR10' else 1 #number of perturbation levels

    return p_list, labels, test_loader, num_levels, c_p_dir

def analyze_corruptions(args):
    p_list, labels, test_loader, num_levels, c_p_dir = get_p_labels(args)
    batch_size = args.batch_size
    stats = np.zeros((len(p_list),3*num_levels), dtype='float32')
    for pi,p in enumerate(p_list):
        print(p)
        dataset = torch.from_numpy(np.float32(np.load(os.path.join(c_p_dir, p + '.npy')).transpose((0,3,1,2))))
        dataset = dataset/255.
        for i in range(num_levels):
            num_test_points = 10000
            dataset_curr = dataset[i*num_test_points:(i+1)*num_test_points]
            labels_curr = labels[i*num_test_points:(i+1)*num_test_points]
            num_batches = len(test_loader)
            for j,(x,y) in enumerate(test_loader):
                data = dataset_curr[(j)*batch_size:(j+1)*batch_size]
                label = labels_curr[j*batch_size:(j+1)*batch_size] 
                # data = un_normalize(data)
                assert((y==label.long()).all())
                delta = (x-data).reshape(batch_size,-1)
                linf_dist = norms_linf_squeezed(delta).mean()
                l1_dist = norms_l1_squeezed(delta).mean()
                l2_dist = norms_l2_squeezed(delta).mean()
                stats[pi,0*num_levels+i] += linf_dist/num_batches
                stats[pi,1*num_levels+i] += l1_dist/num_batches
                stats[pi,2*num_levels+i] += l2_dist/num_batches
    np.set_printoptions(suppress=True)
    
    print(stats)
    return stats        

def test_corruptions(args, model):
    batch_size = args.batch_size  
    p_list, labels, test_loader, num_levels,c_p_dir = get_p_labels(args)

    accuracies = np.zeros((len(p_list),num_levels), dtype='float32')
    classes = np.zeros((len(p_list),3*num_levels), dtype='float32')
    for pi,p in enumerate(p_list):
        print(p)
        dataset = torch.from_numpy(np.float32(np.load(os.path.join(c_p_dir, p + '.npy')).transpose((0,3,1,2))))
        dataset = dataset/255.
        for i in range(num_levels):
            num_test_points = 10000
            dataset_curr = dataset[i*num_test_points:(i+1)*num_test_points]
            labels_curr = labels[i*num_test_points:(i+1)*num_test_points]
            num_batches = 0
            for j in range(num_test_points//batch_size):
                data = dataset_curr[(j)*batch_size:(j+1)*batch_size].to(args.device)
                label = labels_curr[j*batch_size:(j+1)*batch_size].to(args.device)
                preds = model(data)
                acc = (preds.max(1)[1] == label).sum()
                accuracies[pi,0*num_levels] += acc
                
                if args.mode == "pipeline":
                    class_preds = model.forward_classifier(data).max(1)[1]
                    classes[pi,0*num_levels+i] += (class_preds == 0).sum()
                    classes[pi,1*num_levels+i] += (class_preds == 1).sum()
                    classes[pi,2*num_levels+i] += (class_preds == 2).sum()

    accuracies = accuracies/num_test_points
    classes = classes/num_test_points
    np.set_printoptions(suppress=True)
    print(accuracies)
    print(classes)

    return 1

