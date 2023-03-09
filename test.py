from common_corruptions import *
from copy import deepcopy

def test_setup(args):
    device = torch.device("cuda:{0}".format(args.gpu_id) if torch.cuda.is_available() else "cpu")
    print(device)
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
  
    location = args.path
    if location[-1] == "/": location = location[:-1]
    if(not os.path.exists(location)): os.makedirs(location)



    args.device = device
    if args.mode.lower() == "pipeline":
        if args.layer_num == -1: f_e = None
        else: f_e = FeatureExtractor(args); f_e = nn.DataParallel(f_e).to(device)
        p_c = get_perturb_classifier(args)
        
        try:
            p_c.load_state_dict(torch.load(args.path + ".pt", map_location =args.device)); p_c = nn.DataParallel(p_c).to(device)
        except:
            p_c = nn.DataParallel(p_c).to(device); p_c.load_state_dict(torch.load(args.path + ".pt", map_location =args.device))
        
        model = Pipeline(args, f_e, p_c); model = nn.DataParallel(model).to(device)
    
    elif args.mode == "rand":
        layer_nums = []
        p_c_list = nn.ModuleList()
        m_id_list = args.ensemble_id_list
        location += "/rand"
        for i in m_id_list: 
            location += f'_{i}'
        if(not os.path.exists(location)):
            os.makedirs(location)
        for m_id in m_id_list:
            loc = f"{args.path}model_{m_id}/model_info.txt"
            parser = params.parse_args()
            temp_args = parser.parse_args(); temp_args = params.add_params_file(temp_args,loc)
            layer_nums.append(temp_args.layer_num) #This will be used to inform the feature extractor
            p_ci = get_perturb_classifier(temp_args); p_ci = nn.DataParallel(p_ci).to(device)
            p_ci.load_state_dict(torch.load(args.path + f"model_{m_id}/final.pt", map_location =args.device))
            p_c_list.append(p_ci)
        
        layer_nums = set(layer_nums)
        if args.layer_num == -1: f_e = None
        else: f_e = FeatureExtractor(args); f_e = nn.DataParallel(f_e).to(device)
        args.layer_num_list = layer_nums
        model = EnsemblePipeline(args, f_e, p_c_list); model = nn.DataParallel(model).to(device)
    else:
        model = get_model(args)
        if args.model_type != "resnet50": #For resnet50 models, the pip package loads it with the weights.
            try:
                model.load_state_dict(torch.load(args.path + ".pt", map_location = device))
                model = nn.DataParallel(model).to(device)
            except:
                model = nn.DataParallel(model).to(device)
                model.load_state_dict(torch.load(args.path + ".pt", map_location = device))
        else:
            model = nn.DataParallel(model).to(device)


    file = open(f"{location}/test_logs.txt", "a")
    model.eval()
    print(f"Saving attack misclassification at {location}")
    if args.attack == 'corruption':
        analyze_corruptions(args) 
        test_corruptions(args, model) 
    elif args.attack == 'clean': clean_acc(args, model, location)
    elif args.attack == 'pgd': quick_eval(args, model, location)
    elif args.attack == 'auto': test_auto_attack(args, model, location, max_check = 1000, subset = 1, save_images = False)
    else: test_foolbox(args, model, location, max_check = 1000, subset = args.subset)


# python test.py --config configs/MNIST_pipeline.json --num_base 2 --batch_size 1000 --layer_num -1 --use_noise 1 --path models/m_cnn/Static/model_0/final --distance linf
# python test.py --config configs/CIFAR10_pipeline.json --num_base 2 --batch_size 200 --layer_num -1 --use_noise 1 --path models/m_wrn-28-10/Static/model_5/final --dropout 0 --distance linf
# python test.py --config configs/CIFAR10_pipeline.json --num_base 2 --batch_size 200 --layer_num -1 --use_noise 1 --path models/m_wrn-28-10/Static/model_6/final --dropout 0.7 --distance linf
if __name__ == "__main__":
    parser = params.parse_args()
    args = parser.parse_args()
    args = params.add_config(args) if args.config_file != None else args
    print(args)
    test_setup(args)
