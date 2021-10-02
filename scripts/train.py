import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

############ Model Training ########
def train_model(model, dataloader, criterion, optimizer, scheduler, device, epoch, num_epochs):
    model.train()

    running_loss = 0.0
    running_corrects = 0

    loop = tqdm(enumerate(dataloader), total = len(dataloader))
    for i, (inputs,labels) in loop:
        # inputs = inputs.to(device)
        # labels = labels.to(device)

        optimizer.zero_grad()
        if isinstance(labels, (tuple, list)):
            targets1, targets2, lam = labels
            labels = (targets1.to(device), targets2.to(device), lam)
        else:
            labels = labels.to(device)

        outputs = model(inputs)

        loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)

        loss.backward()
        optimizer.step()

        if isinstance(labels, (tuple, list)):
            targets1, targets2, lam = labels
            correct1 = preds.eq(targets1).sum().item()
            correct2 = preds.eq(targets2).sum().item()
            corrects = (lam * correct1 + (1 - lam) * correct2)
        else:
            corrects = torch.sum(preds == labels.data).double()

        loop.set_description(f'epoch: [{epoch}/{num_epochs - 1}]')
        loop.set_postfix(loss = loss.item())
        running_loss += loss.item() * inputs.size(0)
        running_corrects += corrects
        # running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(dataloader.dataset)
    # epoch_acc = running_corrects.double() / len(dataloader.dataset)
    epoch_acc = float(running_corrects) / len(dataloader.dataset)

    return epoch_loss, epoch_acc


############ Model Validating ########
def val_model(model, dataloader, criterion, device, epoch, num_epochs):
    model.eval()

    running_loss = 0.0
    running_corrects = 0

    # Iterate over data.
    loop = tqdm(enumerate(dataloader), total = len(dataloader))
    for i, (inputs,labels) in loop:
        # inputs = inputs.to(device)
        # labels = labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)

        loop.set_description(f'epoch: [{epoch}/{num_epochs - 1}]')
        loop.set_postfix(loss = loss.item())
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects.double() / len(dataloader.dataset)

    return epoch_loss, epoch_acc