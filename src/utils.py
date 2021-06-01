from torchvision.utils import save_image
# import numpy as np
import torch
import os
from torchvision import models
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def saveOutputAsImage(output, path):
    with torch.no_grad():
        save_image(output.float(), path)



if (__name__ == '__main__'):

    torch.cuda.empty_cache()
    NUM_EXAMPLES = 1

    # import model
    model_weights = torch.load('../save/Baseline/baseline1/model_statedict.pt')
    # print ('model: ', model)
    model = models.segmentation.fcn_resnet101(pretrained=True, progress=True)
    model.classifier[4] = nn.Conv2d(512, 2, kernel_size=(1,1), stride=(1,1))
    model.aux_classifier[4] = nn.Conv2d(256, 2, kernel_size=(1,1), stride=(1,1))
    model.load_state_dict(model_weights)
    model.eval()

    # grab a few examples to run through the model
    count = 0
    examples = []
    with os.scandir('../data/deepglobe-dataset-pt/train-labeled-sat') as it:
        for entry in it:
            if not entry.name.startswith('.'):
                if (count == 10):
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

    model.to(device)

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
    
    output = model(batch)['out'].cpu()
    output = torch.argmax(output, dim=1) * 255
    print ('torch.sum(output): ', torch.sum(output))
    #print ('output: ', output)
    plt.imsave('../save/Baseline/baseline1/example_outputs/example_output_' + examples[0] + '.png', output[0])
    print ('saved via plt')





    '''
    # output = output.numpy()
    print ('output.s: ', output.size)
    # im = Image.fromarray(output)
    #print ('output.shape: ', output.shape)
    #print ('output: ', output)
    # im.save('..save/Baseline/baseline0/example_outputs/example_output_' + examples[0])
    predictions = torch.argmax(output, dim=1)
    for i in range(NUM_EXAMPLES):
        cur_example = predictions[i]
        print (cur_example)
        saveOutputAsImage(cur_example, '..save/Baseline/baseline0/example_outputs/example_output_' + examples[i] + '.jpg')
    '''


