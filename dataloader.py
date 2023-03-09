import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import os


normalize_imagenet = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform_imagenet_train = transforms.Compose([transforms.RandomResizedCrop(224),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                normalize_imagenet,])

transform_imagenet_test = transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                normalize_imagenet,])

def get_dataloaders(args, adversarial = False, no_transform = False, return_datasets = False):
    # attacked_model is used for returning the adversarial images dataset. Which model was attacked to generate the new images
    if not adversarial:
        if args.dataset.lower() == "cifar10":
            transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Lambda(lambda x: x.float())])
                                        
            transform_test = transforms.Compose([transforms.ToTensor(),transforms.Lambda(lambda x: x.float())])
            transform_train = transform_test if no_transform else transform_train
            d_train = datasets.CIFAR10("../data", train=True, download=True, transform=transform_train)
            d_test = datasets.CIFAR10("../data", train=False, download=True, transform=transform_test)

            train_loader = DataLoader(d_train, batch_size = args.batch_size, shuffle= not no_transform, num_workers=16)
            test_loader = DataLoader(d_test, batch_size = args.batch_size, shuffle=False, num_workers=16)
        
        elif args.dataset.lower() == "mnist":
            d_train = datasets.MNIST("../data", train=True, download=True, transform=transforms.ToTensor())
            d_test = datasets.MNIST("../data", train=False, download=True, transform=transforms.ToTensor())
            train_loader = DataLoader(d_train, batch_size = args.batch_size, shuffle= not no_transform)
            test_loader = DataLoader(d_test, batch_size = args.batch_size, shuffle=False)
        
        elif args.dataset.lower() == "imagenette":
            transform_train = transforms.Compose([transforms.Resize((128,128)),transforms.RandomCrop(128, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),normalize_imagenet
                                    ])
                                        
            transform_test = transforms.Compose([transforms.Resize((128,128)), transforms.ToTensor(),normalize_imagenet])

            root = "/home/pratyus2/.fastai/data/imagenette2-160"
            traindir = os.path.join(root, 'train')
            valdir = os.path.join(root, 'val')
            train_dataset = datasets.ImageFolder(traindir,transform_train)
            val_dataset = datasets.ImageFolder(valdir,transform_test)


            train_loader = DataLoader(train_dataset, batch_size=args.batch_size,num_workers=4, pin_memory=True, shuffle = True)#sampler=train_sampler
            test_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True, shuffle=False)#sampler=val_sampler,

        
        elif args.dataset.lower() == "imagenet":
            allreduce_batch_size = args.batch_size 
            
            stride = 10
            root = "/home/pratyus2/scratch/data/imagenet"
            traindir = os.path.join(root, 'train')
            valdir = os.path.join(root, 'val')
            train_dataset = StridedImageFolder(traindir,transform_imagenet_train,stride=stride)
            val_dataset = StridedImageFolder(valdir,transform_imagenet_test,stride=stride)


            # train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
            # val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, num_replicas=hvd.size(), rank=hvd.rank())
            train_loader = DataLoader(train_dataset, batch_size=allreduce_batch_size,num_workers=8, pin_memory=True, shuffle = True)#sampler=train_sampler
            test_loader = DataLoader(val_dataset, batch_size=allreduce_batch_size, num_workers=8, pin_memory=True, shuffle=False)#sampler=val_sampler,


    else:
        print ("Adversarial Perturbation Label")
        if args.dataset.lower() in ["cifar10","imagenet","imagenette"]:
            root = f"../data/{args.dataset.upper()}_ADVsmallstep_apgd" 
            root = f"../data/{args.dataset.upper()}_ADVsmallstep" 
            root = f"../data/{args.dataset.upper()}_ADV" 
            # root = f"/home/pratyus2/scratch/projects/multi_adv/data/{args.dataset.upper()}_ADV" 
            
            dim = {"cifar10":32, "imagenette":128, "imagenet":224}[args.dataset.lower()]
            transform_train = transforms.Compose([transforms.ToPILImage(),
                                            transforms.RandomCrop(dim, padding=4),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),normalize_imagenet,
                                            transforms.Lambda(lambda x: x.float())])
            # transform_test = transforms.Compose([transforms.ToPILImage(),transforms.ToTensor(),transforms.Lambda(lambda x: x.float())])
            transform_train = transforms.ToTensor() if args.dataset.lower() == "imagenet" else transform_train
            transform_test = transforms.ToTensor() if args.dataset.lower() == "imagenet" else normalize_imagenet
            dataset_type = AdversarialDatasetFolder if args.dataset.lower() == "imagenet" else AdversarialDataset
            
            d_train = dataset_type(root, args.attack_types, args.attacked_model_list, train = True, transform = transform_train, num_base = args.num_base)
            # train_indices = torch.randperm(len(d_train))[:3000]
            # d_train = Subset(d_train, train_indices)
            train_loader = DataLoader(dataset=d_train, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
            d_test = dataset_type(root, args.attack_types, args.attacked_model_list, train = False, transform = transform_test, num_base = args.num_base)
            test_loader = DataLoader(dataset=d_test, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

        
        elif args.dataset.lower() == "mnist":
            root = "../data/MNIST_ADV"
            d_train = AdversarialDataset(root, args.attack_types, args.attacked_model_list, train = True, transform = None, num_base = args.num_base)
            train_loader = DataLoader(dataset=d_train, batch_size=args.batch_size, shuffle=True)
            d_test = AdversarialDataset(root, args.attack_types, args.attacked_model_list, train = False, transform = None, num_base = args.num_base)
            test_loader = DataLoader(dataset=d_test, batch_size=args.batch_size, shuffle=False)
    if return_datasets:
        return train_loader, test_loader, d_train, d_test
    
    return train_loader, test_loader


class AdversarialDatasetFolder(datasets.ImageFolder):
    def __init__(self, root, attack_types, attacked_model_list, train = True, transform = None, num_base = 3, *args, **kwargs):
        self.new_root = tempfile.mkdtemp()

        train = "train" if train else "test"
        classes = []
        new_classes = []
        idx_to_class = {}
        for i, attack in enumerate(attack_types):
            for model_name in attacked_model_list:
                cls = f"{attack}/{model_name}_x/{train}"
                classes.append(cls)
                new_cls = "_".join(cls.split("/"))
                new_classes.append(new_cls)
                idx_to_class[new_cls] = i

        classes.sort()
        new_classes.sort()
        idx_to_label={}
        for i,cls in enumerate(new_classes):
            idx_to_label[i] = idx_to_class[cls]

        for cls in classes:
            new_cls = "_".join(cls.split("/"))
            os.symlink(os.path.join(root, cls), os.path.join(self.new_root, new_cls), target_is_directory = True)
        
        def target_transform(label):
            label = idx_to_label[label] 
            if num_base == 2: label = min(label,1)
            return label 
                
        super().__init__(self.new_root, target_transform = target_transform, transform = transform)

    def __del__(self):
        shutil.rmtree(self.new_root)

class AdversarialDataset(torch.utils.data.Dataset):
    def __init__(self, root, attack_types, attacked_model_list, train = True, transform = None, num_base = 3):
        train = "train" if train else "test"
        x_list = []; y_list = []
        for class_label, attack in enumerate(attack_types):
            for model_name in attacked_model_list:
                try:
                    x_list.append(torch.load(f"{root}/{attack}/{model_name}_x_{train}.pt"))
                    y_list.append(torch.load(f"{root}/{attack}/{model_name}_y_{train}.pt").long()*0 + class_label)
                except:
                    print(f"No file at: {root}/{attack}/{model_name}_x_{train}.pt. Skipping.")
        self.x_data = torch.cat(x_list)
        self.y_data = torch.cat(y_list)
        torch.manual_seed(0)
        rand=torch.randperm(self.y_data.shape[0]).clone()
        self.x_data = self.x_data[rand]
        self.y_data = self.y_data[rand]
        self.transform = transform
        if num_base == 2:
            if (len(attack_types) == 3): #linf, (l1 l2)
                self.y_data[self.y_data == 2] = 1
            elif (len(attack_types) == 4): #(linf l2 recolor) stadv
                self.y_data[self.y_data < 3] = 0
                self.y_data[self.y_data == 3] = 1
        self.len = self.x_data.shape[0]

    def __getitem__(self, index):
        x_data_index = self.x_data[index]
        if self.transform:
            x_data_index = self.transform(x_data_index)
        return (x_data_index, self.y_data[index])

    def __len__(self):
        return self.len

import tempfile
import shutil

class StridedImageFolder(datasets.ImageFolder):
    def __init__(self, root, *args, **kwargs):
        self.stride = kwargs['stride']
        del kwargs['stride']

        self.new_root = tempfile.mkdtemp()
        classes = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
        classes.sort()
        classes = classes[::self.stride]
        for cls in classes:
            os.symlink(os.path.join(root, cls), os.path.join(self.new_root, cls))

        super().__init__(self.new_root, *args, **kwargs)

    def __del__(self):
        shutil.rmtree(self.new_root)
