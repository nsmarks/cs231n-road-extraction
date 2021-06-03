from torchvision.utils import save_image
import torch
import os
from torchvision import models
import torch.nn as nn
# import numpy as np
# from PIL import Image
import matplotlib.pyplot as plt
import collections
import sys


def saveOutputAsImage(output, path):
    with torch.no_grad():
        save_image(output.float(), path)



if (__name__ == '__main__'):

    torch.cuda.empty_cache()
    NUM_EXAMPLES = 1

    # import model
    model_weights = torch.load('../save/Baseline/baseline1/model_statedict.pt')
    # print ('model: ', model)
    baseline = models.segmentation.fcn_resnet101(pretrained=True, progress=True)
    baseline.classifier[4] = nn.Conv2d(512, 2, kernel_size=(1,1), stride=(1,1))
    baseline.aux_classifier[4] = nn.Conv2d(256, 2, kernel_size=(1,1), stride=(1,1))
    baseline.load_state_dict(model_weights)
    baseline.eval()



    # adversarial:
    # import model
    model_weights = torch.load('../save/Adversarial/adversarial4/model_statedict.pt')
    toRemove = []
    for weight in model_weights:

        # print ('here is a weight: ', weight)
        if "realVsPseudo" in weight:
            toRemove.append(weight)

    for removal in toRemove:
        print ('removing weight')
        del model_weights[removal]

    newNames = []
    vals = []
    newOrderedDict = collections.OrderedDict()
    for weight in model_weights:
        newName = weight[7:]
        newOrderedDict[newName] = model_weights[weight]
    adversarial = models.segmentation.fcn_resnet101(pretrained=True, progress=True)
    adversarial.classifier[4] = nn.Conv2d(512, 2, kernel_size=(1,1), stride=(1,1))
    adversarial.aux_classifier[4] = nn.Conv2d(256, 2, kernel_size=(1,1), stride=(1,1))
    adversarial.load_state_dict(newOrderedDict)
    adversarial.eval()



    if len(sys.argv) != 2:
        print ('Use index in arg')
        quit
    else:
        index = sys.argv[1]

    # grab a few examples to run through the model
    count = 0
    examples = []
    with os.scandir('../data/deepglobe-dataset-pt/train-labeled-sat') as it:
        for entry in it:
            if not entry.name.startswith('.'):
                if (count == index):
                    print (entry.name)
                    examples.append(entry.name)
                else:
                    count += 1
                if (len(examples) == NUM_EXAMPLES):
                    break

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    baseline.to(device)
    adversarial.to(device)


    #batch = torch.cat(examples)
    #batch = batch.reshape(10, 3, 1024, 1024)

    tensors = []

    # save them
    for example in examples:
        tensor = torch.load('../data/deepglobe-dataset-pt/train-labeled-sat/' + example).to(device)
        '''
        output = model(tensor)
        output = output['out']
        # output = output.view(1, output.shape[0], output.shape[1], output.shape[2])
        predictions = torch.argmax(output, dim=0)
        predictions = predictions.reshape(1, predictions.shape[0], predictions.shape[1], predictions.shape[2])
        saveOutputAsImage(predictions, '../save/Baseline/baseline0/example_outputs/example_output_' + example)
        '''
        tensors.append(tensor)

    batch = torch.cat(tensors)
    batch = batch.reshape(NUM_EXAMPLES, 3, 1024, 1024)
    print ('about to run batch')
    
    with torch.no_grad():
        baseline_output = baseline(batch)['out'].cpu()
        baseline_output = torch.argmax(baseline_output, dim=1) * 255
        adversarial_output = adversarial(batch)['out'].cpu()
        adversarial_output = torch.argmax(adversarial_output, dim=1) * 255
    plt.imsave('../save/Baseline/baseline1/example_outputs/baseline_example_output_' + examples[0] + '.png', baseline_output[0])
    plt.imsave('../save/Adversarial/adversarial4/example_outputs/adversarial_example_output_' + examples[0] + '.png', adversarial_output[0])

    print ('saved via plt')



