from util_models import *

def get_p_c(args):
    if args.model_type == "preactresnet":
        p_c = ClassifierResnet(PreActBlock, [2,2,2,2], layer_num = args.layer_num, num_base=args.num_base)
    elif args.model_type == "cnn":
        p_c = ClassifierCNN(layer_num = args.layer_num, num_base=args.num_base, num_features = len(args.feature_models))
    elif args.model_type == "resnet50":
        p_c = models.resnet50(pretrained=True)
        p_c.fc = nn.Linear(2048, args.num_base)
    elif args.model_type == "resnet18":
        p_c = models.resnet18(pretrained = True)
        p_c.fc = nn.Linear(512, args.num_base)
    elif args.model_type == "vgg11":
        p_c = models.vgg11_bn(pretrained = not args.fft)
        p_c.classifier[-1] = nn.Linear(4096, args.num_base)
    elif args.model_type == "cifar-wrn":
        depth = 28
        widen_factor = 10
        p_c = WideResNet(layer_num = -1, depth = depth, widen_factor = widen_factor, dropRate = args.droprate) 
        p_c.load_state_dict(get_un_parallel_dict(torch.load("models/m_wrn-28-10/Base/linf.pt")))
        p_c.fc = nn.Linear(64*widen_factor, args.num_base)
    else:  
        splits = args.model_type.split("-")
        depth = int(splits[1])
        widen_factor = int(splits[2])
        p_c = ClassifierWRN(layer_num = args.layer_num,num_base=args.num_base,
                            depth = depth, widen_factor = widen_factor, 
                            dropRate = args.droprate, num_features = len(args.feature_models))  
    return p_c

def get_perturb_classifier(args):
    if args.fft == 2:
        # In experiments where we want to jointly train a perturbation classifier that uses the fourier represntation and one that does not.
        args_copy = copy.deepcopy(args)
        p_c_fft = PerturbClassifier(args)
        args_copy.fft = 0
        p_c_simple = PerturbClassifier(args_copy)
        p_c = CombPC(args, p_c_simple, p_c_fft)
    else:
        p_c = PerturbClassifier(args)
    return p_c

def get_model(args):
    if args.model_type.lower() == "preactresnet":
            model = PreActResNet18(layer_num = args.layer_num)  
    elif args.model_type.lower() == "cnn":
        model = CNN(layer_num = args.layer_num, dropout = args.dropout)
    elif args.model_type.lower() == "cnn_msd": 
        model = CNN_MSD()
    elif args.model_type.lower() == "cnn_cifar":
        model = CNN_Cifar(layer_num = args.layer_num)
    elif args.model_type.lower() == "resnet50":
        from perceptual_advex.utilities import get_dataset_model
        _, model = get_dataset_model(dataset='cifar',arch='resnet50',checkpoint_fname=args.path + ".pt")
    else:
        splits = args.model_type.split("-")
        depth = int(splits[1])
        widen_factor = int(splits[2])
        model = WideResNet(layer_num = args.layer_num, depth = depth, widen_factor = widen_factor, dropRate = args.dropout)              
    return model

class PerturbClassifier(nn.Module):
    def __init__(self, args):
        super(PerturbClassifier, self).__init__()
        self.args = deepcopy(args)
        self.classifier = get_p_c(args)
        print ("Total model parameters  = ", sum(p.numel() for p in self.parameters() if p.requires_grad))

    def forward(self, f):
        f = get_fft(f) if self.args.fft else f
        xc = self.classifier(f)
        return xc

class FeatureExtractor(nn.Module):
    def __init__(self, args):
        super(FeatureExtractor, self).__init__()
        args = deepcopy(args)
        self.feature_models = get_base_models(args, for_feature_extractor = True)

    def forward(self, x, return_all = False):
        f = [m_i.features(x, return_all = return_all) for m_i in self.feature_models]
        if not return_all:
            f = torch.cat(f, dim = 1) 
            return f
        f = [torch.cat([f[j][i] for j in range(len(f))], dim = 1) for i in range(len(f[0]))]                    
        return f


class EnsemblePipeline(nn.Module):
    def __init__(self, args, f_extractor, p_classifier_list):
        super(EnsemblePipeline, self).__init__()
        self.args = deepcopy(args)
        self.f_e = f_extractor
        self.p_c_list = p_classifier_list
        self.base_models = get_base_models(args)
        hashmap = {'softmax':nn.Softmax(dim = 1), 'sigmoid':nn.Sigmoid(), 'tanh': nn.Tanh(), 'uniform':lambda x : x, 'max':None}
        self.m = hashmap[args.pool] if args.pool != "max" else hashmap['softmax']
        self.layer_num_list = args.layer_num_list
        self.aggregate = "majority"

    def aggregate_return_pc(self, x, f):
        rets = self.list_return_pc(x, f)
        if self.aggregate == "mean":
            return torch.mean(torch.stack(rets), dim = 0)
        else:
            votes = torch.cat([preds.argmax(dim = 1).unsqueeze(1) for preds in rets], dim = 1)
            majority_vote = torch.mode(votes, dim = 1)[0]
            correctness = ((majority_vote.unsqueeze(1) == votes)*2 - 1)
            new_preds = [rets[i]*correctness[:,i].unsqueeze(1) for i in range(len(rets))]
            return torch.mean(torch.stack(new_preds), dim = 0)

    def list_return_pc(self, x, f):
        l_set = self.layer_num_list.copy(); l_set.discard(-1)
        f_is_list = (len(l_set)>1)
        ret = []
        for m in self.p_c_list:
            if m.module.args.features==0: inp = x
            else: inp = f[m.module.args.layer_num] if f_is_list else f
            ret.append(m.module.forward(inp))
        return ret

    def forward_once(self, inp, forward_classifier = False, return_both = False):
        noise = get_noise_like(self.args, inp) if self.args.use_noise else 0
        x = (inp + noise).clamp(0,1)
        #Returning all layers whenever there are atleast two layers apart from -1
        l_set = self.layer_num_list.copy(); l_set.discard(-1)
        f = self.f_e(x, return_all = (len(l_set) > 1)) if (self.layer_num_list != set([-1])) else x 
        xc = self.aggregate_return_pc(x,f)
        if forward_classifier and not return_both: return xc

        bases_ret = [bm(x).unsqueeze(1) for bm in self.base_models]
        x_cat = torch.cat((bases_ret),dim=1)
        if self.args.pool == 'max': ret = x_cat[torch.arange(inp.shape[0]),xc.max(1)[1]]
        elif self.args.pool in ['softmax', 'sigmoid','tanh']:
            x_cat = nn.Softmax(dim = 2)(x_cat) if self.args.probs else x_cat
            xc_pool = self.m(xc)
            ret = torch.bmm(x_cat.permute(0,2,1),xc_pool.unsqueeze(-1)).squeeze(-1)
        elif self.args.pool == 'uniform': ret = torch.mean(x_cat,dim=1)
        return [xc, ret] if return_both else ret

    def forward(self, inp, forward_classifier = False, return_both = False):
        if self.args.use_noise: #We need to do EoT only when we use noise
            ret = [0,0] if return_both else [0]; 
            N_starts = self.args.nEoT
            for _ in range(N_starts):
                curr_ret=self.forward_once(inp, forward_classifier, return_both)
                curr_ret = [curr_ret] if not return_both else curr_ret
                ret = [ret[i] + curr_ret[i] for i in range(len(ret))]
            ret = [ret[i]/float(N_starts) for i in range(len(ret))]
            ret = ret[0] if not return_both else ret
            del curr_ret#; torch.cuda.empty_cache()

        else: ret = self.forward_once(inp, forward_classifier, return_both)
        return ret

class CombPC(nn.Module):
        def __init__(self, args, m1, m2):
            super(CombPC, self).__init__()
            self.args = deepcopy(args)
            self.m1 =m1
            self.m2 = m2
        
        def forward(self, x, forward_one = True, forward_two = True):
            logit = 0
            if forward_one: logit+= self.m1(x)
            if forward_two: logit+= self.m2(x)
            return  logit


class Pipeline(nn.Module):
    def __init__(self, args, f_extractor, p_classifier):
        super(Pipeline, self).__init__()
        self.args = deepcopy(args)
        self.f_e = f_extractor
        self.p_c = p_classifier
        self.base_models = get_base_models(args)
        hashmap = {'softmax':nn.Softmax(dim = 1), 'sigmoid':nn.Sigmoid(), 'tanh': nn.Tanh(), 'uniform':lambda x : x, 'max':None}
        self.m = hashmap[args.pool] if args.pool != "max" else hashmap['softmax']

    def forward_once(self, inp, forward_classifier = False, return_both = False):
        noise = get_noise_like(self.args, inp) if self.args.use_noise else 0
        x = (inp + noise).clamp(0,1)

        f = self.f_e(x) if self.args.features else x
        xc = self.p_c(f)
        if forward_classifier and not return_both: return xc

        bases_ret = [bm(x).unsqueeze(1) for bm in self.base_models] 
        x_cat = torch.cat((bases_ret),dim=1)
        if self.args.pool == 'max': ret = x_cat[torch.arange(inp.shape[0]),xc.max(1)[1]]
        elif self.args.pool in ['softmax', 'sigmoid','tanh']:
            x_cat = nn.Softmax(dim = 2)(x_cat) if self.args.probs else x_cat
            xc_pool = self.m(xc)
            ret = torch.bmm(x_cat.permute(0,2,1),xc_pool.unsqueeze(-1)).squeeze(-1)
        elif self.args.pool == 'uniform': ret = torch.mean(x_cat,dim=1)
        return [xc, ret] if return_both else ret

    def forward(self, inp, forward_classifier = False, return_both = False):
        if self.args.use_noise: #We need to do EoT only when we use noise
            ret = [0,0] if return_both else [0]; 
            N_starts = self.args.nEoT
            for _ in range(N_starts):
                curr_ret=self.forward_once(inp, forward_classifier, return_both)
                curr_ret = [curr_ret] if not return_both else curr_ret
                ret = [ret[i] + curr_ret[i] for i in range(len(ret))]
            ret = [ret[i]/float(N_starts) for i in range(len(ret))]
            ret = ret[0] if not return_both else ret
            del curr_ret; torch.cuda.empty_cache()

        else: ret = self.forward_once(inp, forward_classifier, return_both)
        return ret