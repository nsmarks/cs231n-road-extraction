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

def train_model(model, dataloaders, loss_func, optimizer, num_epochs):
    since = time.time()

    val_iou_history = [] # track validation iou
    val_loss_history = []
    train_iou_history = []
    train_loss_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_iou = 0

    for epoch in range(num_epochs):
        print ()
        print ('Epoch ', epoch, ' of ', num_epochs - 1)
        print ('****************************************')

        for phase in ['train', 'val']: # each epoch trains and then checks val
            if phase == 'train':
                model.train() # set model to training mode: will update weights
            else: # phase == 'val'
                model.eval() # model won't update weights

            running_loss = 0.0 # -------------------------------------------------------------- what's this?

            # TODO TODO TODO running iou

            # iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                print ('got inputs and labels')
                # print ('labels: ', labels)
                # print ('type(labels): ', type(labels))

                # zero param gradients
                optimizer.zero_grad()

                # forward pass - only track history in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    print ('got outputs')
                    # print ('outputs: ', outputs)
                    # print ('type(outputs): ', type(outputs))
                    final_outputs = outputs['out']
                    # TODO --------------------------------------------------------------------------- TODO some sort of output visualizations
                    loss = loss_func(final_outputs, labels)
                    print ('got loss')

                    # compute final predictions
                    predictions = torch.argmax(final_outputs, dim=1)

                    IoU = computeIoU(predictions, labels)

                    # _, preds = torch.max(outpus, 1) # ---- unclear what this does - do we need it?

                    # backward and optimize in train phase
                    if (phase == 'train'):
                        print ('Entering backward')
                        loss.backward()
                        print ('Optimizer step')
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0) # ------------------------------------- not incredibly sure
            
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_iou = IoU / len(dataloaders[phase].dataset)
            print (phase, ' Loss: ', epoch_loss, "\t IoU: ", epoch_iou) # -------------------- IoU

            # copy model if best
            if phase == 'val' and epoch_iou > best_iou:
                best_iou = epoch_iou
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_iou_history.append(epoch_iou)
                val_loss_history.append(epoch_loss)
            if phase == 'train':
                train_iou_history.append(epoch_iou)
                train_loss_history.append(epoch_loss)

    print()

    time_elapsed = time.time() - since
    print ('time_elapsed: ', time_elapsed)
    print ('Training complete in ', time_elapsed // 3600, ' hr, ', time_elapsed // 60, ' min, ', time_elapsed % 60, ' sec')
    print ('Best val IoU: ', best_iou)

    # load best model weights
    model.load_state_dict(best_model_wts)
    history = {
        'val_iou': val_iou_history,
        'val_loss': val_loss_history,
        'train_iou': train_iou_history,
        'train_loss': train_loss_history
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

# create dataloaders
train_dataloader = torch.utils.data.DataLoader(train_labeled_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
dataloader_dict = {'train': train_dataloader, 'val': val_dataloader}


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
loss_func = nn.CrossEntropyLoss()

# train and evaluate
best_model, history = train_model(model, dataloader_dict, loss_func, optimizer, num_epochs=num_epochs)

# TODO: save model and history ---------------------------------------------------------------------------------------------------------

num_files = len(os.listdir('../save/Baseline/'))
os.mkdir('../save/Baseline/baseline' + str(num_files))
torch.save(best_model.state_dict(), '../save/Baseline/baseline' + str(num_files) + '/model_statedict.pt')
w = csv.writer(open('../save/Baseline/baseline' + str(num_files) + '/history.csv', 'w'))
for key, val in history.items():
    w.writerow([key, val])




