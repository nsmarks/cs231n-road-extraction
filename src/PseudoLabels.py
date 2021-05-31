import torch
from torchvision import models
import torch.nn as nn
import torch.optim as optim
import copy
import time
import os
import csv

from TransformedDataset import TransformedDataset

# https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html

def train_model(model, dataloaders, train_unlabeled_dataset, loss_func, optimizer, num_epochs, pseudo_ex_folder):
    since = time.time()

    val_iou_history = [] # track validation iou
    val_loss_history = []
    train_labeled_iou_history = []
    train_labeled_loss_history = []
    train_unlabeled_iou_history = []
    train_unlabeled_loss_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_iou = 0

    
    # epoch 0:
    print ('Epoch 0 of ', num_epochs - 1, '. Training using only labeled train data.')
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
                outputs = model(inputs)
                final_outputs = outputs['out']
                # TODO --------------------------------------------------------------------------- TODO some sort of output visualizations
                loss = loss_func(final_outputs, labels)
                print (phase, ' loss: ', loss)

                # compute final predictions
                predictions = torch.argmax(final_outputs, dim=1)
                # print ('torch.sum(predictions): ', torch.sum(predictions))

                IoU = computeIoU(predictions, labels)
                # print ('IoU: ', IoU)
                runningIoU += IoU

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

    # end of epoch 0
    
    

    # epoch 1 - 9:
    for epoch in range(1, num_epochs):
        print ()
        print ('Epoch ', epoch, ' of ', num_epochs - 1)
        print ('**************************************************************************')
        # create pseudo labels by passing train_unlabeled through model
        train_unlabeled_dataset.has_labels = False
        train_unlabeled_dataset.return_name = True
        train_unlabeled_dataloader = torch.utils.data.DataLoader(train_unlabeled_dataset, batch_size=8, shuffle=False, num_workers=2)
        for inputs, names in train_unlabeled_dataloader:
            # print ('inputs: ', inputs)
            inputs = inputs[0].to(device)
            final_outputs = model(inputs)['out']
            pseudo_labels = torch.argmax(final_outputs, dim=1)
            for i in range(len(pseudo_labels)):
                save_name = names[i][:-7] + '_pseudo_label.pt'
                torch.save(pseudo_labels[i], '../data/deepglobe-dataset-pt/train-pseudo-labels/' + save_name)
                # save progression examples of pseudo labels
                if (save_name == '39512_pseudo_label.pt' or save_name == '74091_pseudo_label.pt'):
                    ex_save_name = save_name[:5] + '_epoch' + str(epoch) + '.pt'
                    torch.save(pseudo_labels[i], pseudo_ex_folder + '/' + ex_save_name)

        train_unlabeled_dataset.has_labels = True
        train_unlabeled_dataset.return_name = False
        dataloaders['train-unlabeled'] = torch.utils.data.DataLoader(train_unlabeled_dataset, batch_size=8, shuffle=True, num_workers=2)

        for phase in ['train-labeled', 'train-unlabeled', 'val']: # each epoch trains and then checks val
            if phase == 'train-labeled' or phase == 'train-unlabeled':
                model.train() # set model to training mode: will update weights
            else: # phase == 'val'
                model.eval() # model won't update weights

            running_loss = 0.0 # -------------------------------------------------------------- what's this?
            runningIoU = 0.0
            # iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                # print ('got inputs and labels')
                # print ('labels: ', labels)
                # print ('type(labels): ', type(labels))

                # zero param gradients
                optimizer.zero_grad()

                # forward pass - only track history in train
                with torch.set_grad_enabled(phase == 'train-labeled' or phase == 'train-unlabeled'):
                    outputs = model(inputs)
                    # print ('got outputs')
                    # print ('outputs: ', outputs)
                    # print ('type(outputs): ', type(outputs))
                    final_outputs = outputs['out']
                    # TODO --------------------------------------------------------------------------- TODO some sort of output visualizations
                    loss = loss_func(final_outputs, labels)
                    # print (phase, ' loss: ', loss)

                    # compute final predictions
                    predictions = torch.argmax(final_outputs, dim=1)
                    # print ('torch.sum(predictions): ', torch.sum(predictions))

                    IoU = computeIoU(predictions, labels)
                    # print ('IoU: ', IoU)
                    runningIoU += IoU

                    # _, preds = torch.max(outpus, 1) # ---- unclear what this does - do we need it?

                    # backward and optimize in train phase
                    if (phase == 'train-labeled' or phase == 'train-unlabeled'):
                        #print ('Entering backward')
                        loss.backward()
                        #print ('Optimizer step')
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0) # ------------------------------------- not incredibly sure

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

def computeIoU(output, label):
    '''Roads labeled as 1, Background labeled as 0'''

    # print ('output: ', output)
    # print ('output.shape: ', output.shape)
    # print ('label: ', label)
    # print ('lable.shape: ', label.shape)
    # road IoU
    road_intersection = (output == label).to(torch.int64) * output * label # an awkward way to do logical and
    road_union = ((output + label) > 0).to(torch.int64)
    roadIoU = (torch.sum(road_intersection) * 1.0) / torch.sum(road_union)

    output_inverse = torch.logical_not(output).to(torch.int64)
    label_inverse = torch.logical_not(label).to(torch.int64)
    background_intersection = (output_inverse == label_inverse).to(torch.int64) * output_inverse * label_inverse
    background_union = ((output_inverse + label_inverse) > 0).to(torch.int64)
    backgroundIoU = (torch.sum(background_intersection) * 1.0) / torch.sum(background_union)

    totalIoU = (roadIoU + backgroundIoU) / 2

    return totalIoU



def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(feature_extract, use_pretrained=True):
    print ('Initializing model...')
    model = models.segmentation.fcn_resnet101(pretrained=use_pretrained, progress=True) # --------------------------which model backbone?
    set_parameter_requires_grad(model, feature_extract)
    # set the number of classes from 21 to 2 (road and not road). All other params are same as default
    model.classifier[4] = nn.Conv2d(512, 2, kernel_size=(1,1), stride=(1,1))
    model.aux_classifier[4] = nn.Conv2d(256, 2, kernel_size=(1,1), stride=(1,1))
    # print ('model: ', model)
    return model


batch_size = 8
num_epochs = 10
feature_extract = True # when False, finetune whole model. When True, only updated reshaped layer

if torch.cuda.is_available():
    print ('[INFO] Moving model and data to cuda')
    device = 'cuda'
else:
    print ('[INFO] Cuda not available')
    device = 'cpu'


model = initialize_model(feature_extract, use_pretrained=True)
model.to(device)


print ('Initializing datasets and dataloaders...')
# create datasets
train_labeled_dataset = TransformedDataset('../data/deepglobe-dataset-pt/train-labeled-sat', '../data/deepglobe-dataset-pt/train-labeled-mask')
val_dataset = TransformedDataset('../data/deepglobe-dataset-pt/val-sat', '../data/deepglobe-dataset-pt/val-mask')
train_unlabeled_dataset = TransformedDataset('../data/deepglobe-dataset-pt/train-unlabeled-sat', '../data/deepglobe-dataset-pt/train-pseudo-labels', has_labels=False, return_name=True)

# create dataloaders
train_dataloader = torch.utils.data.DataLoader(train_labeled_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
# we'll create the train_unlabeled_dataloader during training
dataloader_dict = {'train-labeled': train_dataloader, 'val': val_dataloader}


# create the optimizer
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

optimizer = optim.Adam(params_to_update) # not messing with hyperparams for baseline

# set up loss function
weight = torch.Tensor([1, 15]).to(device) # try to fix the imbalance of background vs road
loss_func = nn.CrossEntropyLoss(weight=weight)




num_files = len(os.listdir('../save/Pseudo/'))
save_folder = '../save/Pseudo/pseudo' + str(num_files)
pseudo_ex_folder = save_folder + '/pseudo-ex'
os.mkdir(save_folder)
os.mkdir(pseudo_ex_folder)


# train and evaluate
best_model, history = train_model(model, dataloader_dict, train_unlabeled_dataset, loss_func, optimizer, num_epochs=num_epochs, pseudo_ex_folder=pseudo_ex_folder)



torch.save(best_model.state_dict(), save_folder + '/model_statedict.pt')
w = csv.writer(open(save_folder + '/history.csv', 'w'))
for key, val in history.items():
    w.writerow([key, val])




