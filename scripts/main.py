from __future__ import print_function
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel, SWALR

import pandas as pd
import argparse
import os
import time
import copy
import pickle

from model import Net
from data import GTSRBDataset, ImbalancedDatasetSampler, CutMixCollator, CutMixCriterion, train_data_transforms, data_transforms
from train import train_model, val_model

from torchsummary import summary

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

parser = argparse.ArgumentParser(description='PyTorch GTSRB example')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=70, metavar='N',
                    help='number of epochs to train (default: 70)')
parser.add_argument('--swa_start', type=int, default=40, metavar='N',
                    help='swa start epoch (default: 40)')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: 0.0001)')
parser.add_argument('--swa_lr', type=float, default=5e-5, metavar='LR',
                    help='swa scheduler learning rate (default: 5e-5)')
parser.add_argument('--cut_mix', type=int, default=0, metavar='N',
                    help='use cut mix: input 1 for use (default: 0)')
parser.add_argument('--order', type=int, default=0, metavar='N',
                    help='model and csv file order for unique naming (default: 0)')
parser.add_argument('--early_stopping', type=int, default=70, metavar='N',
                    help='patience (default: 70)')
args = parser.parse_args()
print()
print(args)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

###### Data Loaders #######
train_dataset = GTSRBDataset(
        img_file_path = os.getcwd() + "/../dataset/train/X.pt", 
        label_file_path = os.getcwd() + "/../dataset/train/y.pt", 
        transform = train_data_transforms,
        device = device,
        )

val_dataset = GTSRBDataset(
    img_file_path =  os.getcwd() + "/../dataset/validation/X.pt", 
    label_file_path = os.getcwd() + "/../dataset/validation/y.pt", 
    transform = data_transforms,
    device = device,
    )

if(args.cut_mix == 1):
     collator = CutMixCollator(1.0)
else:
     collator = None

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size = args.batch_size,
    sampler = ImbalancedDatasetSampler(train_dataset),
    drop_last = True,
    collate_fn=collator,
    )

val_loader = torch.utils.data.DataLoader(
    val_dataset, 
    batch_size = args.batch_size,
    shuffle = False,
    drop_last = True,
    )

num_epochs = args.epochs
swa_start = args.swa_start


###### Model, optimizers and loss function #######

# Model Parameters:
in_channels=3
in_wh=48
fc1_dim=150
num_classes = 43

conv_params = {
     'out_channels':[150,200,300],
     'kernel_size':[7,4,4],
     'stride':[1,1,1],
     'padding':[2,2,2],
     'stn_ch1':[200, 300],
     'stn_ch2': [150, 150],
}

model = Net(in_channels, in_wh, fc1_dim, num_classes, conv_params)
model.to(device)

if(args.cut_mix == 1):
     train_criterion = CutMixCriterion(reduction='mean')
     test_criterion = nn.CrossEntropyLoss(reduction='mean')
else:
     train_criterion = F.nll_loss
     test_criterion = F.nll_loss


optimizer = optim.Adam(model.parameters(), lr = args.lr, betas = (0.9,0.999))
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.1, verbose=True)

### SWA ###
swa_model = AveragedModel(model)
swa_model.to(device)
swa_scheduler = SWALR(optimizer, swa_lr = args.swa_lr)

if torch.cuda.device_count() >= 1:
    print('\nModel pushed to {} GPU(s), type {}.\n'.format(torch.cuda.device_count(), torch.cuda.get_device_name(0)))
else:
    raise ValueError('CPU training is not supported')

summary(model, (3,48,48))

since = time.time()
val_acc_history = []

early_stopping = False
best_val_loss = 1E5
val_no_improve = 0
patience = args.early_stopping
best_acc = 0.0
swa_begin = False
print()
######## Model Train/Validation Phase ########
for epoch in range(swa_start):
     torch.cuda.synchronize()

     if(early_stopping):
          break

     print('Epoch {}/{}'.format(epoch, num_epochs - 1))
     print('-' * 10)

     # Training phase
     train_epoch_loss, train_epoch_acc = train_model(model, train_loader, train_criterion, optimizer, swa_scheduler, device, epoch, num_epochs)
     print('{} Loss: {:.4f} Acc: {:.4f}'.format('Train', train_epoch_loss, train_epoch_acc))

     # Validation phase
     val_epoch_loss, val_epoch_acc = val_model(model, val_loader, test_criterion, device, epoch, num_epochs)
     print('{} Loss: {:.4f} Acc: {:.4f}'.format('Validation', val_epoch_loss, val_epoch_acc))

     if val_epoch_loss < best_val_loss:
          best_acc = val_epoch_acc
          best_val_loss = val_epoch_loss
          model_file = os.getcwd() + '/../checkpoints/model_' + str(args.order) + '.pth'
          torch.save(model.state_dict(), model_file)
          print('Saved model to ' + model_file + '.')
          val_no_improve = 0
     else:
          val_no_improve += 1
     if (patience <= val_no_improve):
          print(f'\nEarly Stopping at epoch {epoch}.')
          early_stopping = True
          break

     val_acc_history.append(val_epoch_acc)
     scheduler.step(round(val_epoch_loss, 2))
     print('\n')

time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
print('Best val Acc: {:4f}'.format(best_acc))
print('Best val Loss: {:4f}'.format(best_val_loss))
print('\n')


######## SWA Phase ########
swa_since = time.time()

for epoch in range(swa_start, num_epochs):
     torch.cuda.synchronize()

     print('Epoch {}/{}'.format(epoch, num_epochs - 1))
     print('-' * 10)

     swa_epoch_loss, swa_epoch_acc = train_model(model, train_loader, train_criterion, optimizer, scheduler, device, epoch, num_epochs)
     print('{} Loss: {:.4f} Acc: {:.4f}'.format('SWA Training', swa_epoch_loss, swa_epoch_acc))

     swa_model.update_parameters(model)
     swa_scheduler.step()
     print()

time_elapsed = time.time() - swa_since
print('SWA complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

# Update bn statistics for the swa_model at the end
torch.optim.swa_utils.update_bn(train_loader, swa_model)

# Use swa_model to make predictions on the validtion data.
print('\nSWA results:')
val_epoch_loss, val_epoch_acc = val_model(swa_model, val_loader, test_criterion, device, num_epochs, num_epochs)

print('SWA val Acc: {:4f}'.format(val_epoch_acc))
print('SWA val Loss: {:4f}'.format(val_epoch_loss))

torch.save(copy.deepcopy(swa_model.state_dict()), os.getcwd() + '/../checkpoints/swa_model_' + str(args.order) + '.pt')
print()

time_elapsed = time.time() - since
print('Experiment complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

outfile = os.getcwd() + '/../checkpoints/gtsrb_kaggle_' + str(args.order) + '.csv'
output_file = open(outfile, "w")
dataframe_dict = {"Filename" : [], "ClassId": []}
test_data = torch.load(os.getcwd() + '/../dataset/testing/test.pt')
file_ids = pickle.load(open(os.getcwd() + '/../dataset/testing/file_ids.pkl', 'rb'))

print('Preparing Kaggle submission file ... ')

with torch.no_grad():
  for i, data in enumerate(test_data):
      data = data_transforms(data.unsqueeze(0))
      data = data.to(device)
      output = swa_model(data)
      pred = output.data.max(1, keepdim=True)[1].item()
      file_id = file_ids[i][0:5]
      dataframe_dict['Filename'].append(file_id)
      dataframe_dict['ClassId'].append(pred)

df = pd.DataFrame(data=dataframe_dict)
df.to_csv(outfile, index=False)
print("Written to csv file {}".format(outfile))





