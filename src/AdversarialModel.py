# from typing import Sequence
from json import load
import torch
import torch.nn as nn
from torchvision import models
import RoadUtils
import random


class AdversarialModel(nn.Module):
    def __init__(self, feature_extract, train_labeled_dataloader, device, load_weights_loc):
        super().__init__()
        self.train_labeled_dataloader = train_labeled_dataloader
        self.device = device

        print ('Initializing resnet')
        self.resnet = models.segmentation.fcn_resnet101(pretrained=True, progress=True)
        RoadUtils.set_parameter_requires_grad(self.resnet, feature_extract)
        # set the number of classes from 21 to 2 (road and not road). All other params are same as default
        self.resnet.classifier[4] = nn.Conv2d(512, 2, kernel_size=(1,1), stride=(1,1))
        self.resnet.aux_classifier[4] = nn.Conv2d(256, 2, kernel_size=(1,1), stride=(1,1))
        if (load_weights_loc is not None):
            resnet_weights = torch.load(load_weights_loc)
            self.resnet.load_state_dict(resnet_weights)

        self.realVsPseudoClassifier = RealVsPseudoClassifier()

    def forward(self, inputs, phase):
        outputs = self.resnet(inputs)
        final_outputs = outputs['out']
        if (phase in ['train-labeled', 'val', 'test']):
            return final_outputs
        else:
            print ('entering train-unlabeled in AdversarialModel')
            # we're in train-unlabeled, so apply adversarial discriminator to outputs
            # get random true labels from train-labeled
            generated_masks = torch.argmax(final_outputs, dim=1)
            num_outputs = final_outputs.shape[0]
            true_labels = []
            for _train_labeled_inputs, train_labeled_labels in self.train_labeled_dataloader:
                print ('train_labeled_labels: ', train_labeled_labels)
                for train_labeled_label in train_labeled_labels:
                    train_labeled_label = train_labeled_label.to(self.device)
                    true_labels.append(train_labeled_label)
                    if len(true_labels) == num_outputs:
                        break
                if (len(true_labels) == num_outputs):
                    break
            zeros = [0] * num_outputs # 0 means generated mask
            ones = [1] * num_outputs # 1 means real mask
            label_order = (zeros + ones) # concatenate. This is label for discriminator
            random.shuffle(label_order)
            print ('label_order: ', label_order)
            masks_for_discriminator = []
            real_used = 0
            generated_used = 0
            for mask_label in label_order:
                if mask_label == 0: # generated
                    masks_for_discriminator.append(generated_masks[generated_used])
                    generated_used += 1
                else: # mask_label == 1
                    masks_for_discriminator.append(true_labels[real_used])
                    real_used += 1
            # batch1 = torch.FloatTensor(masks_for_discriminator[:num_outputs])
            # batch2 = torch.FloatTensor(masks_for_discriminator[num_outputs:])

            batch1 = torch.zeros(num_outputs, 1024, 1024) # TODO Unclear if this works
            batch2 = torch.zeros(num_outputs, 1024, 1024)

            for i in range(num_outputs):
                batch1[i] = masks_for_discriminator[i]
                batch2[i] = masks_for_discriminator[i + num_outputs]

            batch1 = batch1.reshape(num_outputs, 1, 1024, 1024)
            batch2 = batch2.reshape(num_outputs, 1, 1024, 1024)

            print ('batch1: ', batch1)
            print ('batch2: ', batch2)
            adversarialOutput1 = self.realVsPseudoClassifier(batch1)
            adversarialOutput2 = self.realVsPseudoClassifier(batch2)

            print ('adversarialOutput1: ', adversarialOutput1)
            print ('adversarialOutput2: ', adversarialOutput2)
            output = torch.cat((adversarialOutput1, adversarialOutput2))
            print ('output: ', output)
            return output, torch.Tensor(label_order).long()










    
class GradReverse(torch.autograd.Function):
    """Gradient reversal layer as defined by Sato et al. in
    "Adversarial Training for Cross-Domain Universal Dependency Parsing."
    In forward pass, act as identity transform.
    In backward pass, multiply gradient by lambda.
    Structure based on https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html
    and https://discuss.pytorch.org/t/solved-reverse-gradients-in-backward-pass/3589/3.
    """

    @staticmethod
    def forward(ctx, input):
        """ctx is unneeded context object. input is tensor."""
        return input

    @staticmethod
    def backward(ctx, grad_output):
        """grad_output is gradient of loss wrt output"""
        #TODO: Should lambda be negative? Unclear in paper
        lambdaVar = -0.5
        return lambdaVar * grad_output

class RealVsPseudoClassifier(nn.Module):
    """ Branch of network classifying whether label is 
    real or pseudo label.
    """
    def __init__(self):
        super().__init__()
        # TODO TODO
        self.sequential = nn.Sequential(
            nn.Conv2d(1, 5, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=5),
            nn.Dropout(),
            nn.Conv2d(5, 10, 8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=5),
            nn.Dropout(),
            nn.Flatten(1),
            nn.Linear(15210, 2) # TODO TODO fix the input dim
        )
        self.gradReverse = GradReverse.apply

    def forward(self, label):
        gradReversed = self.gradReverse(label)
        output = self.sequential(gradReversed)
        return output