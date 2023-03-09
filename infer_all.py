import numpy as np
import argparse
from glob import glob

parser = argparse.ArgumentParser(description='Draw inference from Results', formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--dataset", help = "MNIST/CIFAR 10/CIFAR-nonlp", type = str, default = "MNIST")  # MNIST, CIFAR10
parser.add_argument("--num_samples", help = "Number of Test Examples", type = int, default = 1000)
parser.add_argument("--path", help = "Override model (custom path)", type = str, default = None)
parser.add_argument("--pool", help = "Override model (custom path)", type = str, default = 'max')
parser.add_argument("--result", help = "Just values", type = int, default = 0)

params = parser.parse_args()
num_samples = params.num_samples
dataset = params.dataset
path = params.path
folder = path
pool = params.pool
result_mode = params.result

attacks_list_1 = [f'{pool}_l1',f'{pool}_l1fab-t']

# attacks_list_inf = [f'{pool}_FGSM', f'{pool}_linf', f'{pool}_PGD',f'{pool}_IGM',f'{pool}_linfapgd-ce', f'{pool}_linfapgd-dlr', f'{pool}_linfapgd-t',f'{pool}_linffab', f'{pool}_linffab-t', f'{pool}_linfsquare']
attacks_list_inf = [f'{pool}_linf',f'{pool}_linfapgd-ce', f'{pool}_linfapgd-dlr', f'{pool}_linfapgd-t', f'{pool}_linffab-t', f'{pool}_linfsquare']

# attacks_list_2 = [f'{pool}_l2',f'{pool}_IGD',f'{pool}_AGNA',f'{pool}_DeepFool', f'{pool}_DDN',f'{pool}_CWL2',f'{pool}_l2apgd-ce', f'{pool}_l2apgd-dlr', f'{pool}_l2apgd-t',f'{pool}_l2fab', f'{pool}_l2fab-t', f'{pool}_l2square']
attacks_list_2 = [f'{pool}_l2',f'{pool}_ddn',f'{pool}_l2apgd-ce', f'{pool}_l2apgd-dlr', f'{pool}_l2apgd-t', f'{pool}_l2fab-t', f'{pool}_l2square']



out = open("extras/" + dataset +"_RES/" + folder.split("/")[-1] + "test_logs.txt", "w")

def myprint(s):
    print(s)
    out.write(str(s) + "\n")


files = glob(folder + "/*.*")
not_found = []
attacks_npy_1 = []
attacks_npy_2 = []
attacks_npy_inf = []

l1_attacks = np.ones((num_samples, len(attacks_list_1)))
l2_attacks = np.ones((num_samples, len(attacks_list_2)))
linf_attacks = np.ones((num_samples, len(attacks_list_inf)))

all_attacks = np.ones((num_samples, 3))
pall_attacks = np.ones((num_samples, 3))

if dataset != "CIFAR-nonlp":
    for a in attacks_list_1:
        try:
            y = np.load(folder + "/" + a+ ".npy")
            attacks_npy_1.append(y.reshape(1000))
        except:
            y = np.load(folder + "/" +  attacks_list_1[-1] + ".npy")
            attacks_npy_1.append(y.reshape(1000))
            not_found.append(a)

else:
    stadv= np.load(f"{folder}/{pool}_stadv.npy")
    recolor= np.load(f"{folder}/{pool}_recolor.npy")
    
for a in attacks_list_2:
    try:
        y = np.load(folder + "/" + a+ ".npy")
        attacks_npy_2.append(y.reshape(1000))
    except:
        y = np.load(folder + "/" +  attacks_list_2[0] + ".npy")
        attacks_npy_2.append(y.reshape(1000))
        not_found.append(a)

for a in attacks_list_inf:
    try:
        y = np.load(folder + "/" + a+ ".npy")
        attacks_npy_inf.append(y.reshape(1000))
    except:
        y = np.load(folder + "/" +  attacks_list_inf[0] + ".npy")
        attacks_npy_inf.append(y.reshape(1000))
        not_found.append(a)
    

if dataset == "MNIST": e_l1 = 10; e_l2 = 2.0; e_linf = 0.3 
else: e_l1 = 10; e_l2 = 0.5; e_linf = 0.03 


linf_dict = {'attacks_npy': attacks_npy_inf, 'attacks_list': attacks_list_inf, 'e': e_linf}
l1_dict = {'attacks_npy': attacks_npy_1, 'attacks_list': attacks_list_1, 'e': e_l1}
l2_dict = {'attacks_npy': attacks_npy_2, 'attacks_list': attacks_list_2, 'e': e_l2}

def print_func(name, acc):
    if not result_mode:
        myprint(name + " : " + str(acc))
    else:
        print(f"{acc:.1f}\%")


def get_acc(pos, l_dict):
    a = l_dict['attacks_npy'][pos]
    name = l_dict['attacks_list'][pos]
    accuracy = (1-(a.sum()/num_samples)) *100
    print_func(name,accuracy)
    return a



for i in range(len(attacks_list_inf)):
    linf_attacks[:,i] = get_acc(i,linf_dict)

linf_min = np.sum(linf_attacks, axis = 1).astype("bool")
accuracy = (1-(linf_min.sum()/num_samples)) *100
print_func("linf", accuracy)


for i in range(len(attacks_list_2)):
    l2_attacks[:,i] = get_acc(i,l2_dict)
l2_min = np.sum(l2_attacks, axis = 1).astype("bool")
accuracy = (1-(l2_min.sum()/num_samples)) *100
print_func("l2", accuracy)

if dataset != "CIFAR-nonlp":
    for i in range(len(attacks_list_1)):
        l1_attacks[:,i] = get_acc(i,l1_dict)

    l1_min = np.sum(l1_attacks, axis = 1).astype("bool")
    accuracy = (1-(l1_min.sum()/num_samples)) *100
    print_func("l1", accuracy)
else:
    print_func("stadv", (1-(stadv.sum()/num_samples)) *100)
    print_func("recolor", (1-(recolor.sum()/num_samples)) *100)
    nonlp_min = (recolor+stadv)[:,0]


all_attacks[:,0] = l1_min if dataset != "CIFAR-nonlp" else nonlp_min
all_attacks[:,1] = l2_min
all_attacks[:,2] = linf_min

all_min = np.sum(all_attacks, axis = 1).astype("bool")
accuracy = (1-(all_min.sum()/num_samples)) *100
print_func("All", accuracy)

print(not_found)

