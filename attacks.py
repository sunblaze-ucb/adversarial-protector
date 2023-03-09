import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random, sys
sys.path.append("./auto-attack/")
sys.path.append("./auto-attack/autoattack")
from autoattack import AutoAttack
from autopgd_base import APGDAttack
from square import SquareAttack

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

def loss_crossentropy(x_adv, x_natural, y, model, distance = None, dataset = "CIFAR10"):
    if dataset.lower() == "imagenet": x_adv = ImagenetTransform(x_adv)
    preds = model(x_adv)
    loss = nn.CrossEntropyLoss()(preds, y)
    return loss

def get_dual_preds(f_e, p_c, x, num_attacks, both = False):
    f, bases_ret = f_e(x, final = True)
    attack_type = p_c(f)
    bases_ret = [x_i.unsqueeze(1) for x_i in bases_ret]
    bases_ret = [bases_ret[0], bases_ret[1]] if num_attacks == 2 else bases_ret
    x_cat = torch.cat((bases_ret),dim=1)
    classes = attack_type.max(1)[1]
    preds = x_cat[torch.arange(x.shape[0]),classes]
    if both:
        return preds, attack_type
    return preds

def loss_dual(params, x_adv, y, y_aux, model_list):
    lamb = 10
    batch_size = x_adv.shape[0]
    assert(isinstance(model_list, list))
    if params.num_attacks == 2:
        y_aux = min(y_aux, 1) #only 2 attacks, l_2 should be marked as 1
    y_aux = torch.ones_like(y)*y_aux
    f_e = model_list[0]
    p_c = model_list[1]
    preds, attack_type = get_dual_preds(f_e, p_c, x_adv, params.num_attacks, both = True)
    loss = nn.CrossEntropyLoss()(preds, y) + lamb*nn.CrossEntropyLoss()(attack_type, y_aux) 
    return loss



def square_linf(model, X, y, params, train_mode = False):
    return square(model, X,y, params, train_mode, "Linf", params.epsilon_l_inf)

def square_l1(model, X, y, params, train_mode = False):
    return square(model, X,y, params, train_mode, "L1", params.epsilon_l_1)

def square_l2(model, X, y, params, train_mode = False):
    return square(model, X,y, params, train_mode, "L2", params.epsilon_l_2)

def square(model, X, y, params, train_mode, norm, epsilon):
    if not isinstance(model, list):
        is_training = model.training
        if not train_mode:
            model.eval()    # Need to freeze the batch norm and dropouts unles specified not to
    else:
        is_training = model[1].training
        if not train_mode:
            model[1].eval()    # Need to freeze the batch norm and dropouts unles specified not to
    
    adversary = SquareAttack(model, p_init=.8, n_queries = params.num_iter, norm = norm, n_restarts = 1, eps = epsilon, verbose = False)
    delta = adversary.perturb(X.clone(), y) - X


    if is_training:
        try:
            model.train()    #Reset to train mode if model was training earlier
        except:
            model[1].train()
    return delta


def apgd_linf(model, X, y, params, train_mode = False):
    return apgd(model, X,y, params, train_mode, "Linf", params.epsilon_l_inf)

def apgd_l1(model, X, y, params, train_mode = False):
    return apgd(model, X,y, params, train_mode, "L1", params.epsilon_l_1)

def apgd_l2(model, X, y, params, train_mode = False):
    return apgd(model, X,y, params, train_mode, "L2", params.epsilon_l_2)

def apgd(model, X, y, params, train_mode, norm, epsilon):
    if not isinstance(model, list):
        is_training = model.training
        if not train_mode:
            model.eval()    # Need to freeze the batch norm and dropouts unles specified not to
    else:
        is_training = model[1].training
        if not train_mode:
            model[1].eval()    # Need to freeze the batch norm and dropouts unles specified not to
    
    adversary = APGDAttack(model, n_iter = params.num_iter, norm = norm, n_restarts = 1, eps = epsilon, verbose = False)
    delta = adversary.perturb(X.clone(), y) - X


    if is_training:
        try:
            model.train()    #Reset to train mode if model was training earlier
        except:
            model[1].train()
    return delta

def pgd_linf(model, X, y, params, train_mode = False):
    if params.dataset.lower() == "imagenet": X = InverseImagenetTransform(X)
    if not isinstance(model, list):
        is_training = model.training
        if not train_mode:
            model.eval()    # Need to freeze the batch norm and dropouts unles specified not to
    else:
        is_training = model[1].training
        if not train_mode:
            model[1].eval()    # Need to freeze the batch norm and dropouts unles specified not to
    epsilon = params.epsilon_l_inf
    alpha = params.alpha_l_inf
    num_iter = params.num_iter
    restarts = params.restarts
    randomize = params.randomize
    smallest_adv = params.smallest_adv
    criterion = loss_crossentropy 

    criterion = loss_dual if isinstance(model,list) else criterion

    if randomize == 2:
        randomize = np.random.randint(2) if restarts == 1 else 1 
    #If there are more than 1 restarts, anyways the following loop ensures that atleast one of the starts is from 0 when rand = 1
    
    assert(restarts>=1)
    if alpha == None:
        alpha = epsilon * 0.01/0.3
   
    max_delta = torch.zeros_like(X, requires_grad=False).cpu()
    
    for i in range (restarts):
        delta = torch.empty_like(X).uniform_(-epsilon, epsilon)
        delta.requires_grad = True
        if i==0 and (randomize==0 or restarts > 1): 
            #Make a 0 initialization if you are making multiple restarts
            #or if explicitly told not to randomize for a single start
            delta = torch.zeros_like(X, requires_grad=True)    
        loss = 0
        for t in range(num_iter):
            if smallest_adv: 
                output = model(X+delta)
                incorrect = output.max(1)[1] != y 
                correct = (~incorrect).unsqueeze(1).unsqueeze(1).unsqueeze(1) 
            else :
                correct = 1.
            #Finding the correct examples so as to attack only them            
            # loss = nn.CrossEntropyLoss()(model(X + delta), y)
            loss = criterion(X+delta, X, y, model, distance = "linf", dataset = params.dataset) if criterion is not loss_dual else loss_dual(params, X+delta, y, 0, model)
            loss.backward()
            grads = delta.grad.detach()
            grads[grads!= grads] = 0 #To set nans to zero
            delta.data = (delta.data + alpha*correct*grads.sign()).clamp(-epsilon,epsilon)
            delta.data = torch.min(torch.max(delta.detach(), -X), 1-X) # clip X+delta to [0,1]
            delta.grad.zero_()

        if not isinstance(model, list):
            output = model(X+delta)
            incorrect = output.max(1)[1] != y
            #Edit Max Delta only for successful attacks  
            if i==0:
                max_delta = delta.detach().cpu()
            else:
                max_delta[incorrect] = delta.detach()[incorrect].cpu()
        else:
            output, incorrect = 0, 0
            max_delta = delta.detach().cpu()
        del delta, loss, output, incorrect
        torch.cuda.empty_cache()

    if is_training:
        try:
            model.train()    #Reset to train mode if model was training earlier
        except:
            model[1].train()
    if params.dataset.lower() == "imagenet": max_delta /= imagenet_std
    return max_delta.to(X.device)


def stadv(model, X, y, params):
    from perceptual_advex.attacks import StAdvAttack, ReColorAdvAttack, JPEGLinfAttack

    is_training = model.training
    model.eval()    #Need to freeze the batch norm and dropouts
    attack_stadv = StAdvAttack(model,bound=0.05)
    delta = attack_stadv(X,y) - X
    if is_training: model.train()    #Reset to train mode if model was training earlier
    return delta

def recolor(model, X, y, params):
    from perceptual_advex.attacks import StAdvAttack, ReColorAdvAttack, JPEGLinfAttack

    is_training = model.training
    model.eval()    #Need to freeze the batch norm and dropouts
    attack_recolor = ReColorAdvAttack(model,bound=0.06)
    delta = attack_recolor(X,y) - X
    if is_training: model.train()    #Reset to train mode if model was training earlier
    return delta

def jpeg(model, X, y, params):
    from perceptual_advex.attacks import StAdvAttack, ReColorAdvAttack, JPEGLinfAttack

    is_training = model.training
    model.eval()    #Need to freeze the batch norm and dropouts
    attack_jpeg = JPEGLinfAttack(model,"cifar",bound=0.25)
    delta = attack_jpeg(X,y) - X
    if is_training: model.train()    #Reset to train mode if model was training earlier
    return delta

def ddn(model, X, y, params):
    from fast_adv.attacks import DDN
    is_training = model.training
    model.eval()    #Need to freeze the batch norm and dropouts
    epsilon = params.epsilon_l_2
    ddn_attacker = DDN(steps=100, device=X.device)
    delta_l_2_ddn = ddn_attacker.attack(model, X, labels=y, targeted=False) - X
    delta_l_2_ddn.data *=  epsilon / norms(delta_l_2_ddn.detach()).clamp(min=epsilon) 
    if is_training: model.train()    #Reset to train mode if model was training earlier
    return delta_l_2_ddn


def pgd_l2(model, X, y, params, train_mode = False, CONST=1e-6):
    if params.dataset.lower() == "imagenet": X = InverseImagenetTransform(X)
    try:
        is_training = model.training
        if not train_mode:
            model.eval()    # Need to freeze the batch norm and dropouts unles specified not to
    except:
        is_training = model[1].training
        if not train_mode:
            model[1].eval() 
    epsilon = params.epsilon_l_2
    alpha = params.alpha_l_2
    num_iter = params.num_iter
    restarts = params.restarts
    randomize = params.randomize
    smallest_adv = params.smallest_adv
    criterion = loss_crossentropy
    criterion = loss_dual if isinstance(model,list) else criterion

    if randomize == 2:
        randomize = np.random.randint(2) if restarts == 1 else 1 
    #If there are more than 1 restarts, anyways the following loop ensures that atleast one of the starts is from 0 when rand = 1
    
    assert(restarts>=1)
    max_delta = torch.zeros_like(X, requires_grad=False).cpu()
    for i in range (restarts):
        delta = torch.rand_like(X, requires_grad=True) 
        delta.data *= (2.0*delta.data - 1.0)
        delta.data = delta.data*epsilon/(norms_l2(delta.detach()) + CONST)
        if i==0 and (randomize==0 or restarts > 1): 
            #Make a 0 initialization if you are making multiple restarts
            #or if explicitly told not to randomize for a single start
            delta = torch.zeros_like(X, requires_grad=True)  
        loss = 0

        for t in range(num_iter):
            if smallest_adv: 
                output = model(X+delta)
                incorrect = output.max(1)[1] != y 
                correct = (~incorrect).unsqueeze(1).unsqueeze(1).unsqueeze(1) 
            else :
                correct = 1.
            #Finding the correct examples so as to attack only them            
            loss = criterion(X+delta, X, y, model, distance = "l2", dataset = params.dataset) if criterion is not loss_dual else loss_dual(params, X+delta, y, 2, model)
            loss.backward()
            grads = delta.grad.detach()
            grads[grads != grads] = 0 #To set nans to 0
            delta.data +=  correct*alpha*grads / (norms_l2(grads) + CONST)
            delta.data *= epsilon / norms_l2(delta.detach()).clamp(min=epsilon)
            delta.data = torch.min(torch.max(delta.detach(), -X), 1-X) # clip X+delta to [0,1]
            delta.grad.zero_()  

        if not isinstance(model, list):
            output = model(X+delta)
            incorrect = output.max(1)[1] != y
            #Edit Max Delta only for successful attacks  
            if i==0:
                max_delta = delta.detach().cpu()
            else:
                max_delta[incorrect] = delta.detach()[incorrect].cpu()
        else:
            output, incorrect = 0, 0
            max_delta = delta.detach().cpu()
        del delta, loss, output, incorrect
        torch.cuda.empty_cache()

    if is_training:
        try:
            model.train()    #Reset to train mode if model was training earlier
        except:
            model[1].train()
    if params.dataset.lower() == "imagenet": max_delta /= imagenet_std
    return max_delta.to(X.device)  




def pgd_l1(model, X, y, params, train_mode = False, CONST=1e-6):
    if params.dataset.lower() == "imagenet": X = InverseImagenetTransform(X)
    try:
        is_training = model.training
        if not train_mode:
            model.eval()    # Need to freeze the batch norm and dropouts unles specified not to
    except:
        is_training = model[1].training
        if not train_mode:
            model[1].eval() 
    epsilon = params.epsilon_l_1
    alpha_l_1 = params.alpha_l_1
    num_iter = params.num_iter
    restarts = params.restarts 
    randomize = params.randomize
    smallest_adv = params.smallest_adv
    gap = params.gap
    k = params.k
    criterion = loss_crossentropy
    criterion = loss_dual if isinstance(model,list) else criterion
    if randomize == 2:
        randomize = np.random.randint(2) if restarts == 1 else 1 
    #If there are more than 1 restarts, anyways the following loop ensures that atleast one of the starts is from 0 when rand = 1
  
    assert(restarts>=1)
    #Gap : Dont attack pixels closer than the gap value to 0 or 1
    
    max_delta = torch.zeros_like(X, requires_grad=False).cpu()
    alpha = alpha_l_1/float(k)

    for i in range(restarts):
        delta = torch.from_numpy(np.random.laplace(size=X.shape)).float().to(X.device)
        delta.data = delta.data*epsilon/(norms_l1(delta.detach()) + CONST)
        delta.requires_grad = True
        if i==0 and (randomize==0 or restarts > 1): 
            #Make a 0 initialization if you are making multiple restarts
            #or if explicitly told not to randomize for a single start
            delta = torch.zeros_like(X, requires_grad=True)  
        loss = 0

        for t in range (num_iter):
            if params.k == 1000:
                k = random.randint(5,100)
                alpha = alpha_l_1/float(k)
            if smallest_adv: 
                output = model(X+delta)
                incorrect = output.max(1)[1] != y 
                correct = (~incorrect).unsqueeze(1).unsqueeze(1).unsqueeze(1) 
            else :
                correct = 1.
            # Finding the correct examples so as to attack only them            
            loss = criterion(X+delta, X, y, model, distance = "l1", dataset = params.dataset) if criterion is not loss_dual else loss_dual(params, X+delta, y, 0, model)
            loss.backward()
            grads = delta.grad.detach()
            grads[grads!= grads] = 0 #To set nans to zero
            delta.data += alpha*correct*l1_dir_topk(grads, delta.data, X, gap, k)
            if (norms_l1(delta) > epsilon).any():
                delta.data = proj_l1ball(delta.data, epsilon)
            delta.data = torch.min(torch.max(delta.detach(), -X), 1-X) # clip X+delta to [0,1] 
            delta.grad.zero_() 

        if not isinstance(model, list):
            output = model(X+delta)
            incorrect = output.max(1)[1] != y
            #Edit Max Delta only for successful attacks  
            if i==0:
                max_delta = delta.detach().cpu()
            else:
                max_delta[incorrect] = delta.detach()[incorrect].cpu()
        else:
            output, incorrect = 0, 0
            max_delta = delta.detach().cpu()
        del delta, loss, output, incorrect
        torch.cuda.empty_cache()

    if is_training:
        try:
            model.train()    #Reset to train mode if model was training earlier
        except:
            model[1].train()
    if params.dataset.lower() == "imagenet": max_delta /= imagenet_std
    return max_delta.to(X.device)

def kthlargest(tensor, k, dim=-1):
    val, idx = tensor.topk(k, dim = dim)
    return val[:,:,-1], idx[:,:,-1]

def l1_dir_topk(grad, delta, X, gap, k = 10) :
    #Check which all directions can still be increased such that
    #they haven't been clipped already and have scope of increasing
    X_curr = X + delta
    batch_size = X.shape[0]
    channels = X.shape[1]
    pix = X.shape[2]
    neg1 = (grad < 0)*(X_curr <= gap)
    neg2 = (grad > 0)*(X_curr >= 1-gap)
    neg3 = X_curr < 0
    neg4 = X_curr > 1
    neg = neg1 + neg2 + neg3 + neg4
    u = neg.view(batch_size,1,-1)
    grad_check = grad.view(batch_size,1,-1)
    grad_check[u] = 0

    kval = kthlargest(grad_check.abs().float(), k, dim = 2)[0].unsqueeze(1)
    k_hot = (grad_check.abs() >= kval).float() * grad_check.sign()
    return k_hot.view(batch_size, channels, pix, pix)


def proj_l1ball(x, epsilon=10):
    assert epsilon > 0
    # compute the vector of absolute values
    u = x.abs()
    if (u.sum(dim = (1,2,3)) <= epsilon).all():
        # print (u.sum(dim = (1,2,3)))
         # check if x is already a solution
        return x

    # v is not already a solution: optimum lies on the boundary (norm == s)
    # project *u* on the simplex
    y = proj_simplex(u, s=epsilon)
    # compute the solution to the original problem on v
    y *= x.sign()
    y *= epsilon/norms_l1(y)
    return y


def proj_simplex(v, s=1):
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    batch_size = v.shape[0]
    # check if we are already on the simplex    
    '''
    #Not checking this as we are calling this from the previous function only
    if v.sum(dim = (1,2,3)) == s and np.alltrue(v >= 0):
        # best projection: itself!
        return v
    '''
    # get 'the' array of cumulative sums of a sorted (decreasing) copy of v
    u = v.view(batch_size,1,-1)
    n = u.shape[2]
    u, indices = torch.sort(u, descending = True)
    cssv = u.cumsum(dim = 2)
    # get the number of > 0 components of the optimal solution
    vec = u * torch.arange(1, n+1).float().to(v.device)
    comp = (vec > (cssv - s)).float()

    u = comp.cumsum(dim = 2)
    w = (comp-1).cumsum(dim = 2)
    u = u + w
    rho = torch.argmax(u, dim = 2)
    rho = rho.view(batch_size)
    c = torch.FloatTensor([cssv[i,0,rho[i]] for i in range( cssv.shape[0]) ]).to(v.device)
    c = c-s
    # compute the Lagrange multiplier associated to the simplex constraint
    theta = torch.div(c,(rho.float() + 1))
    theta = theta.view(batch_size,1,1,1)
    # compute the projection by thresholding v using theta
    w = (v - theta).clamp(min=0)
    return w


def SAT(model, X, y, params, train_mode = False, CONST=1e-6):
    attack = random.choice([pgd_l1, pgd_l2, pgd_linf])
    return attack(model, X, y, params, train_mode, CONST)

def norms(Z):
    return Z.view(Z.shape[0], -1).norm(dim=1)[:,None,None,None]

def norms_l2(Z):
    return norms(Z)

def norms_l2_squeezed(Z):
    return norms(Z).squeeze(1).squeeze(1).squeeze(1)

def norms_l1(Z):
    return Z.view(Z.shape[0], -1).abs().sum(dim=1)[:,None,None,None]

def norms_l1_squeezed(Z):
    return Z.view(Z.shape[0], -1).abs().sum(dim=1)[:,None,None,None].squeeze(1).squeeze(1).squeeze(1)

def norms_l0(Z):
    return ((Z.view(Z.shape[0], -1)!=0).sum(dim=1)[:,None,None,None]).float()

def norms_l0_squeezed(Z):
    return ((Z.view(Z.shape[0], -1)!=0).sum(dim=1)[:,None,None,None]).float().squeeze(1).squeeze(1).squeeze(1)

def norms_linf(Z):
    return Z.view(Z.shape[0], -1).abs().max(dim=1)[0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

def norms_linf_squeezed(Z):
    return Z.view(Z.shape[0], -1).abs().max(dim=1)[0]


