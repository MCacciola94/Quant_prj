import resnet
from torch.nn.utils import prune
import torch
import torchvision.models as models

model_names = sorted(name for name in resnet.__dict__
    if name.islower() and not name.startswith("__")
                     and name.startswith("resnet")
                     and callable(resnet.__dict__[name]))

available_arcs= model_names + ["resnet18", "resnet50"]

def is_available(name):
    return name in available_arcs

def load_arch(name, resume = "", already_pruned = True):
    if not(is_available(name)):
        print("Architecture requested not available")
        return None


    if name in model_names:
        model = torch.nn.DataParallel(resnet.__dict__[name]())
    else:
        model = models.__dict__[name]()
    model.cuda()

    



    if not(resume == "")  and already_pruned:
        for m in model.modules(): 
            if hasattr(m, 'weight'):
                pruning_par=[((m,'weight'))]

                if hasattr(m, 'bias') and not(m.bias==None):
                    pruning_par.append((m,'bias'))

                prune.global_unstructured(pruning_par, pruning_method=at.ThresholdPruning, threshold=1e-18)

                
    # optionally resume from a checkpoint

    if resume:
        if os.path.isfile(resume):
            print("=> loading checkpoint '{}'".format(resume))
            checkpoint = torch.load(resume)
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                .format(checkpoint['best_prec1'], checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(resume))

    return model