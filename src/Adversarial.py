import torch
from torchvision import models
import torch.nn as nn
import torch.optim as optim
import copy
import time
import os
import csv
import RoadUtils
from AdversarialModel import AdversarialModel
import sys

from TransformedDataset import TransformedDataset

# https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html

def train_model(model, dataloaders, seg_loss_func, discrim_loss_func, optimizer, num_epochs, num_labeled_only_epochs):
    since = time.time()

    val_iou_history = [] # track validation iou
    val_loss_history = []
    train_labeled_iou_history = []
    train_labeled_loss_history = []
    train_unlabeled_iou_history = []
    train_unlabeled_loss_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_iou = 0

    # epoch 0 to (num_labeled_only_epochs-1):
    for epoch in range(num_labeled_only_epochs):
        print ('Epoch ', epoch, ' of ', num_epochs - 1, '. Training using only labeled train data.')
        print ('**************************************************************************')
        for phase in ['train-labeled', 'val']: # epoch trains and then checks val
            if phase == 'train-labeled':
                model.train() # set model to training mode: will update weights
            else: # phase == 'val'
                model.eval() # model won't update weights

            running_loss = 0.0
            runningIoU = 0.0
            # iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero param gradients
                optimizer.zero_grad()

                # forward pass - only track history in train
                with torch.set_grad_enabled(phase == 'train-labeled'):
                    final_outputs = model(inputs, phase)
                    # final_outputs = outputs['out']
                    # TODO --------------------------------------------------------------------------- TODO some sort of output visualizations
                    loss = seg_loss_func(final_outputs, labels)
                    print (phase, ' loss: ', loss)

                    # compute final predictions
                    predictions = torch.argmax(final_outputs, dim=1)
                    # print ('torch.sum(predictions): ', torch.sum(predictions))

                    IoU = RoadUtils.computeIoU(predictions, labels) # individual IoU for each example in batch
                    batchIoU = torch.sum(IoU)
                    # print ('IoU: ', IoU)
                    runningIoU += batchIoU

                    # backward and optimize in train phase
                    if (phase == 'train-labeled'):
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0) # ------------------------------------- not incredibly sure
        
            # end of phase in epoch 0:
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_iou = runningIoU / len(dataloaders[phase].dataset)
            print (phase, ' Loss: ', epoch_loss, "\t IoU: ", epoch_iou) # -------------------- IoU

            # copy model if best
            if phase == 'val' and epoch_iou > best_iou:
                best_iou = epoch_iou
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_iou_history.append(epoch_iou)
                val_loss_history.append(epoch_loss)
            if phase == 'train-labeled':
                train_labeled_iou_history.append(epoch_iou)
                train_labeled_loss_history.append(epoch_loss)
                train_unlabeled_iou_history.append(0) # we don't use unlabeled data in epoch 0
                train_unlabeled_loss_history.append(0)

        # end epochs of only labeled data
    
    

    # epoch (num_labeled_only_epochs) - (num_epochs-1):
    for epoch in range(num_labeled_only_epochs, num_epochs):
        print ()
        print ('Epoch ', epoch, ' of ', num_epochs - 1, '. Using all data.')
        print ('**************************************************************************')

        for phase in ['train-labeled', 'train-unlabeled', 'val']: # each epoch trains and then checks val

            if phase == 'train-labeled' or phase == 'train-unlabeled':
                model.train() # set model to training mode: will update weights
            else: # phase == 'val'
                model.eval() # model won't update weights

            running_loss = 0.0
            runningIoU = 0.0
            # iterate over data
            for inputs, info in dataloaders[phase]:
                if (phase == 'train-unlabeled'):
                    inputs = inputs[0].to(device) # not entirely sure why it's stored in a list of len one in this case
                else:
                    inputs = inputs.to(device)

                if (phase == 'train-labeled' or phase == 'val'):
                    labels = info
                    labels = labels.to(device)
                

                # zero param gradients
                optimizer.zero_grad()

                # forward pass
                with torch.set_grad_enabled(phase == 'train-labeled' or phase == 'train-unlabeled'):
                    final_outputs = model(inputs, phase)
                    # print ('got outputs')
                    # print ('outputs: ', outputs)
                    # print ('type(outputs): ', type(outputs))
                    # final_outputs = outputs['out']
                    # TODO --------------------------------------------------------------------------- TODO some sort of output visualizations

                    if (phase == 'train-labeled' or phase == 'val'):
                        loss = seg_loss_func(final_outputs, labels)
                        # print (phase, ' loss: ', loss)

                        # compute final predictions
                        predictions = torch.argmax(final_outputs, dim=1)
                        # print ('torch.sum(predictions): ', torch.sum(predictions))

                        IoU = RoadUtils.computeIoU(predictions, labels) # individual IoU for each example in batch
                        batchIoU = torch.sum(IoU)
                        # print ('IoU: ', IoU)
                        runningIoU += batchIoU
                    else: # phase == 'train-unlabeled
                        discrim_output, label_order = final_outputs
                        loss = discrim_loss_func(discrim_output, label_order)
                    
                    # backward and optimize in train phase
                    if (phase == 'train-labeled' or phase == 'train-unlabeled'):
                        #print ('Entering backward')
                        loss.backward()
                        #print ('Optimizer step')
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0) # ------------------------------------- not incredibly sure

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_iou = runningIoU / len(dataloaders[phase].dataset) # should be divided by number of epochs ___________________________________________ TODO TODO
            print (phase, ' Loss: ', epoch_loss, "\t IoU: ", epoch_iou) # -------------------- IoU

            # copy model if best
            if phase == 'val' and epoch_iou > best_iou:
                best_iou = epoch_iou
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_iou_history.append(epoch_iou)
                val_loss_history.append(epoch_loss)
            elif phase == 'train-labeled':
                train_labeled_iou_history.append(epoch_iou)
                train_labeled_loss_history.append(epoch_loss)
            elif phase == 'train-unlabeled':
                train_unlabeled_iou_history.append(epoch_iou)
                train_unlabeled_loss_history.append(epoch_loss)
        # at end of epoch, check for early stopping
        if (epoch > 3):
            recent_loss = val_loss_history[-1]
            if (recent_loss > val_loss_history[-2] and recent_loss > val_loss_history[-3] and recent_loss > val_loss_history[-4]):
                print ('Stopping early!')
                break


    print()

    time_elapsed = time.time() - since
    print ('time_elapsed: ', time_elapsed)
    print ('Training complete in ', time_elapsed // 60, ' min, ', time_elapsed % 60, ' sec')
    print ('Best val IoU: ', best_iou)

    # load best model weights
    model.load_state_dict(best_model_wts)
    history = {
        'val_iou': val_iou_history,
        'val_loss': val_loss_history,
        'train_labeled_iou': train_labeled_iou_history,
        'train_labeled_loss': train_labeled_loss_history,
        'train_unlabeled_iou': train_unlabeled_iou_history,
        'train_unlabeled_loss': train_unlabeled_loss_history
    }
    return model, history

# end train_model


num_labeled_only_epochs = 1
batch_size = 8
num_epochs = 10
feature_extract = True # when False, finetune whole model. When True, only updated reshaped layer

if torch.cuda.is_available():
    print ('[INFO] Setting device to cuda')
    device = 'cuda'
else:
    print ('[INFO] Cuda not available')
    device = 'cpu'


print ('Initializing datasets and dataloaders...')
# create datasets
train_labeled_dataset = TransformedDataset('../data/deepglobe-dataset-pt/train-labeled-sat', '../data/deepglobe-dataset-pt/train-labeled-mask', small_set=True)
val_dataset = TransformedDataset('../data/deepglobe-dataset-pt/val-sat', '../data/deepglobe-dataset-pt/val-mask', small_set=True)
train_unlabeled_dataset = TransformedDataset('../data/deepglobe-dataset-pt/train-unlabeled-sat', '../data/deepglobe-dataset-pt/train-pseudo-labels', has_labels=False, return_name=True, small_set=True)

# create dataloaders
train_labeled_dataloader = torch.utils.data.DataLoader(train_labeled_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
train_unlabeled_dataloader =  torch.utils.data.DataLoader(train_unlabeled_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

dataloader_dict = {'train-labeled': train_labeled_dataloader, 'val': val_dataloader, 'train-unlabeled': train_unlabeled_dataloader}


if (sys.argv > 1):
    load_weights_loc = sys.argv[1]
else:
    load_weights_loc = None

model = AdversarialModel(feature_extract, train_labeled_dataloader, device, load_weights_loc)
model.to(device)

# create the optimizer ________________________________ TODO TODO must account for discriminator
params_to_update = model.parameters()
if (feature_extract):
    params_to_update = []
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print ('\t', name)
else: # not features_extract
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            print ('\t', name)

optimizer = optim.Adam(params_to_update) # TODO: hyperparams

# set up loss function
weight = torch.Tensor([1, 15]).to(device) # try to fix the imbalance of background vs road
seg_loss_func = nn.CrossEntropyLoss(weight=weight)
discrim_loss_func = nn.CrossEntropyLoss()




num_files = len(os.listdir('../save/Adversarial/'))
save_folder = '../save/Adversarial/adversarial' + str(num_files)
# pseudo_ex_folder = save_folder + '/pseudo-ex'
os.mkdir(save_folder)
# os.mkdir(pseudo_ex_folder)


# train and evaluate
best_model, history = train_model(model, dataloader_dict, seg_loss_func, discrim_loss_func, optimizer, num_epochs=num_epochs, num_labeled_only_epochs=num_labeled_only_epochs)



torch.save(best_model.state_dict(), save_folder + '/model_statedict.pt')
w = csv.writer(open(save_folder + '/history.csv', 'w'))
for key, val in history.items():
    w.writerow([key, val])




