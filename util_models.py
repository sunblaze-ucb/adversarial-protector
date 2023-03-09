import torch, sys, numpy as np, random
import torch.nn as nn
sys.path.append("./models_src/")
from preactresnet import *
from wideresnet import *
from cnn import *
from cnn_cifar import *
from attacks import *
from fft import get_fft
from copy import deepcopy
import copy
from torchvision import models
# For Loading Pretrained Models

def get_un_parallel_dict(d):
    # When we use Data parallel -- any model module {model.X} get wrapped under {model.module.X}
    # Removing the "module." from the name while loading to match the correct keys.
    new_d = {}
    for key in d.keys():
        new_d[key[7:]] = d[key]
    return new_d

def freeze_model(args, model):
    model.eval()
    for model_params in model.parameters():
        model_params.requires_grad = False
    if args.fine_tune_last_layer: #This has no meaning for the feature extractor. Only for the base models.
        if args.model_type[0] == 'w':
            model.fc.weight.requires_grad = True
            model.fc.bias.requires_grad = True
        elif args.model_type == "cnn":
            model.classifier.fc3.weight.requires_grad = True
            model.classifier.fc3.bias.requires_grad = True
        else:
            raise("Not Implemented")
    return model

def get_base_models(args, for_feature_extractor = False):
    #To simplify the code, now only supporting 1 particular model type for each perturbation specific robust predictor.
    if not for_feature_extractor: 
        if(args.num_base != len(args.attacked_model_list)):
            print("WARNING: Using only first two attacked models")
            base_names = args.attacked_model_list[:2]
        else:
            base_names = args.attacked_model_list
        
    base_names = args.feature_models if for_feature_extractor else base_names
    base_models = nn.ModuleList()
    layer_num = args.layer_num if for_feature_extractor else -1
    if args.dataset == "MNIST":
        for id in base_names:
            model_name = f"models/m_cnn/Base/{id}"
            model = CNN(layer_num = layer_num).cuda()
            d = get_un_parallel_dict(torch.load(f"{model_name}.pt"))
            model.load_state_dict(d)
            base_models.append(model)
    elif args.dataset == "CIFAR10":
        for id in base_names:
            if id in ["stadv", "recolor", "jpeg"]:
                from perceptual_advex.utilities import get_dataset_model
                m_map = {"stadv":"stadv_0.1.pt","recolor":"recoloradv_0.06.pt"}
                _, model = get_dataset_model(dataset='cifar',arch='resnet50',checkpoint_fname=f'../data/cifar/{m_map[id]}')
                base_models.append(model)
            else:
                model_name = f"models/m_wrn-28-10/Base/{id}"
                model = WideResNet(layer_num = layer_num, depth = 28, widen_factor = 10, dropRate = args.dropout).cuda() 
                model.load_state_dict(get_un_parallel_dict(torch.load(f"{model_name}.pt")))
                base_models.append(model)
    for model in base_models:
        freeze_model(args, model)
    return base_models

def get_noise_like(args, inp):
    def l1_noise(inp, mag):
        noise_1 = torch.from_numpy(np.random.laplace(size=inp.shape)).float().to(inp.device) 
        noise_1 *= mag/norms_l1(noise_1)
        return noise_1

    def l2_noise(inp, mag):
        noise_2 = torch.normal(0, 0.25, size=inp.shape).cuda()
        noise_2 *= mag/norms_l2(noise_2)
        return noise_2

    def linf_noise(inp, mag):
        noise_inf = torch.empty_like(inp).uniform_(-mag,mag)
        return noise_inf

    noise_list = []
    mapper = {"l1":l1_noise, "l2":l2_noise, "linf":linf_noise}
    # mag_2 = 0.5 if dataset == "CIFAR10"  else 2.0
    # mag_1 = 5
    # mag_inf = 0.001 if dataset == "CIFAR10" else 0.1
    for i in range(len(args.noise_list)):
        func = mapper[args.noise_list[i]]
        noise_list.append(func(inp, args.noise_mag_list[i]))

    return random.choice(noise_list)