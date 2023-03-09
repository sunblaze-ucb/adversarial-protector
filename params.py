import argparse
from distutils import util
import yaml
import sys

def parse_args():
    parser = argparse.ArgumentParser(description='Adversarial Training')
    ## Basics
    parser.add_argument("--config_file", help="Configuration file containing parameters", type=str)
    parser.add_argument("--dataset", help="MNIST/CIFAR10", type=str, default = "CIFAR10")
    parser.add_argument("--mode", help="pipeline/standard", type=str, default = "pipeline")
    parser.add_argument("--model_type", help="cnn/wrn-40-2/wrn-28-10/preactresnet", 
                            type=str, default = "preactresnet")# choices = ["cnn","cnn_msd","vit_b_16","wrn-10-2","wrn-10-1","wrn-16-1","wrn-16-2","wrn-40-2","wrn-28-10","wrn-70-16","preactresnet","resnet50","resnet18"])
    parser.add_argument("--gpu_id", help="Id of GPU to be used", type=int, default = 0)
    parser.add_argument("--distance", help="Type of Adversarial Perturbation", type=str)#, choices = ["linf", "l1", "l2", "vanilla"])
    parser.add_argument("--batch_size", help = "Batch Size for Train Set (Default = 128)", type = int, default = 128)
    parser.add_argument("--model_id", help = "For Saving", type = str, default = 0)
    parser.add_argument("--seed", help = "Seed", type = int, default = 0)
    parser.add_argument("--randomize", help = "For the individual attacks", type = int, default = 1, choices = [0,1,2])
    
    #Only valid if restarts = 1: #0 -> Always start from Zero, #1-> Always start with 1, #2-> Start from 0/rand with prob = 1/2
    parser.add_argument("--device", help = "To be assigned later", type = str, default = 'cuda:1')
    parser.add_argument("--epochs", help = "Number of Epochs", type = int, default = 30)
    
    parser.add_argument("--custom_loss", help = "Softmax for Pipelined loss", type = int, default = 0, choices = [0,1])
    parser.add_argument("--use_noise", help = "Random Noise", type = int, default = 0, choices = [0,1])
    
    parser.add_argument("--perturb_stats", help = "Use any specially changed dataset", type = int, default = 0)
    parser.add_argument("--fine_tune_last_layer", help = "Fine tune last layer", type = int, default = 0)
    
    parser.add_argument("--fft", help = "Fast Fourier Transform", type = int, default = 0)

    #For Static Training
    parser.add_argument("--club_l1_l2", help = "How many types of attacks should be sent?", type = int, default = 1, choices = [0,1])
    parser.add_argument("--num_base", help = "How many base models", type = int, default = 2, choices = [1,2,3,4,5,6])
    parser.add_argument('--attack_types', nargs='+', default=["linf","l1","l2"])
    parser.add_argument('--attacked_model_list', nargs='+', default = ["vanilla"])
    parser.add_argument('--feature_models', nargs='+', default = ["vanilla"])
    parser.add_argument('--noise_list', nargs='+', default = [])
    parser.add_argument('--noise_mag_list', nargs='+', type=float, default = [])
    


    #Lp Norm Dependent
    parser.add_argument("--alpha_l_1", help = "Step Size for L1 attacks", type = float, default = 1.0)
    parser.add_argument("--alpha_l_2", help = "Step Size for L2 attacks", type = float, default = 0.01)
    parser.add_argument("--alpha_l_inf", help = "Step Size for Linf attacks", type = float, default = 0.8/255.)
    parser.add_argument("--num_iter", help = "PGD iterations", type = int, default = 50)
    parser.add_argument("--epsilon_l_1", help = "Step Size for L1 attacks", type = float, default = 10)
    parser.add_argument("--epsilon_l_2", help = "Epsilon Radius for L2 attacks", type = float, default = 0.5)
    parser.add_argument("--epsilon_l_inf", help = "Epsilon Radius for Linf attacks", type = float, default = 8/255.)
    parser.add_argument("--restarts", help = "Random Restarts", type = int, default = 1)
    parser.add_argument("--smallest_adv", help = "Early Stop on finding adv", type = int, default = 0)
    parser.add_argument("--gap", help = "For L1 attack", type = float, default = 0.05)
    parser.add_argument("--k", help = "For L1 attack", type = int, default = 10)
    
    #LR
    parser.add_argument("--dropout", help = "Dropout in the model", type = float, default = 0)
    parser.add_argument("--lr_mode", help = "Step wise or Cyclic", type = int, default = 1)
    parser.add_argument("--opt_type", help = "Optimizer", type = str, default = "SGD")
    parser.add_argument("--lr_max", help = "Max LR", type = float, default = 1e-3)
    parser.add_argument("--lr_min", help = "Min LR", type = float, default = 0.)

    #Resume
    parser.add_argument("--resume", help = "For Resuming from checkpoint", type = int, default = 0)
    parser.add_argument("--resume_iter", help = "Epoch to resume from", type = int, default = -1)
    
    
    #For Classifier Based Methods Only
    parser.add_argument("--target", help = "Loss from Classifier/Pipeline", type = str, default = 'classifier', choices = ['both', 'classifier', 'pipeline'])
    parser.add_argument("--pool", help = "max/softmax/sigmoid", type = str, default = "softmax", choices = ['max','softmax','sigmoid','tanh','uniform'])
    parser.add_argument("--generate_one", help = "Generate only 1 type of attack in one batch", type = int, default = 1, choices = [0,1])
    parser.add_argument("--momentum", help = "momentum", type = float, default = 1)
    parser.add_argument("--features", help = "Use Features", type = int, default = 1, choices = [0,1,2])
    parser.add_argument("--layer_num", help = "which layer features to use", type = int, default = -1, choices = [-1,0,1,2,3,4])
    parser.add_argument("--factor", help = "Avg Pool in preact resnet or not", type = int, default = 1)
    parser.add_argument("--staged_train", help = "Static Adv Examples created stage wise", type = int, default = 0, choices = [0,1])
    
    
    parser.add_argument("--attack_vanilla", help = "Pipeline", type = int, default = 0, choices = [0,1])
    
    parser.add_argument("--base_path", help = "Path for Vanilla Model", type = str)
    parser.add_argument("--zeta", help = "softmax power", type = int, default = 1)
    parser.add_argument("--disentangle", help = "SNNL factor", type = float, default = -10)
    parser.add_argument("--probs", help = "Add over the probabilty outputs rather than confidence values", type = int, default = 0)
    parser.add_argument("--droprate", help = "Droprate for Classifier only", type = float, default = 0)
    

    #TEST
    parser.add_argument("--path", help = "Path for test model load", type = str, default = None)
    parser.add_argument("--adaptive", help = "Adaptive Softmax transfer adversary", type = int, default = 0)
    parser.add_argument("--subset", help = "For Foolbox", type = int, default = 0)
    parser.add_argument("--attack", help = "Choose type", type = str, default = 'auto', choices = ['clean','auto', 'pgd', 'foolbox', 'corruption'])
    parser.add_argument("--auto_specific", help = "Auto type", type = str, default = None)
    parser.add_argument('--ensemble_id_list', nargs='+', default = ["105"])
    parser.add_argument('--layer_num_list', nargs='+', default = [])
    parser.add_argument('--nEoT', help = "Randomization averaging", type = int, default = 1)

    return parser

def add_config(args):
    data = yaml.full_load(open(args.config_file,'r'))
    args_dict = args.__dict__
    for key, value in data.items():
        if('--'+key in sys.argv and args_dict[key] != None): ## Giving higher priority to arguments passed in cli
            continue
        if isinstance(value, list):
            args_dict[key] = []
            args_dict[key].extend(value)
        else:
            args_dict[key] = value
    args.features = (args.layer_num != -1)
    return args

def add_params_file(args,loc):
    data = yaml.full_load(open(loc,'r'))
    args_dict = args.__dict__
    for key, value in data.items():
        if isinstance(value, list):
            args_dict[key] = []
            args_dict[key].extend(value)
        else:
            args_dict[key] = value
    args.features = (args.layer_num != -1)
    return args