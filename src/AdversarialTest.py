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

def test_model(model, test_dataloader):
    runningIoU = 0.0

    for inputs, labels in test_dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        final_outputs = model(inputs, 'test')
        predictions = torch.argmax(final_outputs, dim=1)
        IoU = RoadUtils.computeIoU(predictions, labels)
        batchIoU = torch.sum(IoU)
        runningIoU += batchIoU

    total_iou = runningIoU / len(test_dataloader.dataset)
    print ('IoU: ', total_iou)
        


batch_size = 8
if torch.cuda.is_available():
    print ('[INFO] Setting device to cuda')
    device = 'cuda'
else:
    print ('[INFO] Cuda not available')
    device = 'cpu'


print ('Initializing datasets and dataloaders...')
# create datasets
test_dataset = TransformedDataset('../data/deepglobe-dataset-pt/test-sat', '../data/deepglobe-dataset-pt/test-mask', small_set=True)

# create dataloaders
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=2)

if (len(sys.argv) != 2):
    print ('Wrong usage. Give location of weights to load.')
else:
    load_weights_loc = sys.argv[1]

# import model
model_weights = torch.load(load_weights_loc)
# print ('model: ', model)
model = models.segmentation.fcn_resnet101(pretrained=True, progress=True)
model.classifier[4] = nn.Conv2d(512, 2, kernel_size=(1,1), stride=(1,1))
model.aux_classifier[4] = nn.Conv2d(256, 2, kernel_size=(1,1), stride=(1,1))
model.load_state_dict(model_weights)
model.eval()
model.to(device)

num_files = len(os.listdir('../save/Adversarial/'))
save_folder = '../save/Adversarial/adversarialTest' + str(num_files)
os.mkdir(save_folder)

test_model(model, test_dataloader)









