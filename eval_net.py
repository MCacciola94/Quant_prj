import torch
import torch.nn as nn
import argparse
import architectures as archs
import quik_pruning as qp
from trainer import Trainer
import data_loaders as dl
import aux_tools as at
import resnet




def par_count(model):
    res = 0
    for m in model.modules():
        if isinstance(m,nn.Conv2d):
            dims = m.weight.shape
            res += dims[0]*dims[1]*dims[2]*dims[3]
        if isinstance(m,nn.Linear):
            dims = m.weight.shape
            res += dims[0]*dims[1]
    return res

def compress_block(block):
    if isinstance(block,resnet.BasicBlock):
        



parser = argparse.ArgumentParser(description='evaluation of pruned net')
parser.add_argument('--name',
                    help="Name of checkpoint")

args = parser.parse_args()
model = archs.load_arch("resnet20", 10)
qp.prune_thr(model,1.e-12)
base_checkpoint=torch.load("saves/save_" + args.name +"/checkpoint.th")
model.load_state_dict(base_checkpoint['state_dict'])
dataset = dl.load_dataset("Cifar10", 128)
# define loss function (criterion) and optimizer
criterion = nn.CrossEntropyLoss().cuda()


optimizer = torch.optim.SGD(model.parameters(), 0.1,
                            momentum=0.9,
                            weight_decay=5.e-4)

lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                    milestones=[300], last_epoch= - 1)


trainer = Trainer(model = model, dataset = dataset, reg = None, lamb = 1.0, threshold = 0.05, 
                                        criterion =criterion, optimizer = optimizer, lr_scheduler = lr_scheduler, save_dir = "./delete_this_folder", save_every = 44, print_freq = 100)

trainer.validate(reg_on = False)
_, a = at.sparsityRate(model)
print(a)
print(par_count(model))