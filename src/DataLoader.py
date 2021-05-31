import matplotlib.pyplot as plt

import torch
from torchvision import transforms

import pandas as pd
import numpy as np
from PIL import Image

# some inspiration taken from:
# https://medium.com/jun-devpblog/pytorch-4-custom-dataset-class-convert-image-to-tensor-and-vice-versa-57afd90a4313
# https://www.kaggle.com/balraj98/road-extraction-from-satellite-images-deeplabv3#data
# https://medium.com/codex/saving-and-loading-transformed-image-tensors-in-pytorch-f37b4daa9658
# https://stackoverflow.com/questions/65447992/pytorch-how-to-apply-the-same-random-transformation-to-multiple-image


class RoadsDataset():
    '''
    Take in given dataset from DeepGlobe and create pytorch files of data
    with custom splits.
    Possible addition: vary custom splits
    '''

    def __init__(self, root_dir=None, augmentation=None):
        '''
            root_dir to dataset
                test (no masks)
                valid (no masks)
                train (images and masks)
                metadata.csv
        '''
        
        # if root_dir passed in, read in dataset and save as pt tensors
        if (root_dir is not None):
            print ('*******************************************************************')
            print ('Original dataset directory provided. Will read and save as tensors.')


            self.toTensor = transforms.ToTensor()
            self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            
            metadata = pd.read_csv(root_dir + '/metadata.csv')

            train_metadata = metadata[metadata['split'] == 'train']
            # shuffle train data
            train_metadata = train_metadata.sample(frac=1).reset_index(drop=True)
            train_metadata_labeled = train_metadata[:1000]
            new_test_metadata = train_metadata[1000:2101]
            new_val_metadata = train_metadata[2101:3344]
            train_metadata_unlabeled_from_train = train_metadata[3344:]
            old_val_metadata = metadata[metadata['split'] == 'valid']
            old_test_metadata = metadata[metadata['split'] == 'test']
            train_metadata_unlabeled = pd.concat([train_metadata_unlabeled_from_train, old_val_metadata, old_test_metadata])
            # shuffle new unlabeled train
            train_metadata_unlabeled = train_metadata_unlabeled.sample(frac=1).reset_index(drop=True)

            new_dataset_dir = '../data/deepglobe-dataset-pt2'
            print ('new_dataset_dir: ', new_dataset_dir)

            print ('Creating train-labeled tensors...')

            # train-labeled
            for index, labeled_train_elem in train_metadata_labeled.iterrows():
                image = Image.open(root_dir + '/' + labeled_train_elem['sat_image_path'])
                # print ('image: ', image)
                mask = Image.open(root_dir + '/' + labeled_train_elem['mask_path'])
                # print ('mask: ', mask)
                transformed_image, transformed_mask = augmentation([image, mask])
                # print ('transformed_mask.shape: ', transformed_mask.shape)
                one_hot_transformed_mask = self.one_hot_mask(transformed_mask)
                # print ('one_hot_transformed_mask.shape: ', one_hot_transformed_mask.shape)
                # print ('one_hot_transformed_mask: ', one_hot_transformed_mask)

                # print (labeled_train_elem)
                image_id = labeled_train_elem['image_id']


                torch.save(transformed_image, new_dataset_dir + '/train-labeled-sat/' + str(image_id) + '_sat.pt')
                torch.save(one_hot_transformed_mask, new_dataset_dir + '/train-labeled-mask/' + str(image_id) + '_mask.pt')

            print ('Creating train-unlabeled tensors...')

            # train-unlabeled
            for index, unlabeled_train_elem in train_metadata_unlabeled.iterrows():
                image = Image.open(root_dir + '/' + unlabeled_train_elem['sat_image_path'])
                transformed_image = augmentation([image])
                image_id = unlabeled_train_elem['image_id']
                torch.save(transformed_image, new_dataset_dir + '/train-unlabeled-sat/' + str(image_id) + '_sat.pt')

            print ('Creating val tensors...')

            
            # val
            for index, val_elem in new_val_metadata.iterrows():
                image = Image.open(root_dir + '/' + val_elem['sat_image_path'])
                mask = Image.open(root_dir + '/' + val_elem['mask_path'])

                # no augmentation, just to tensor
                transformed_image = self.toTensor(image)
                transformed_image = self.normalize(transformed_image)
                transformed_mask = self.toTensor(mask)

                one_hot_transformed_mask = self.one_hot_mask(transformed_mask)

                image_id = val_elem['image_id']

                torch.save(transformed_image, new_dataset_dir + '/val-sat/' + str(image_id) + '_sat.pt')
                torch.save(one_hot_transformed_mask, new_dataset_dir + '/val-mask/' + str(image_id) + '_mask.pt')

            print ('Creating test tensors...')


            # test
            for index, test_elem in new_test_metadata.iterrows():
                image = Image.open(root_dir + '/' + test_elem['sat_image_path'])
                mask = Image.open(root_dir + '/' + test_elem['mask_path'])

                # no augmentation, just to tensor
                transformed_image = self.toTensor(image)
                transformed_image = self.normalize(transformed_image)
                transformed_mask = self.toTensor(mask)

                one_hot_transformed_mask = self.one_hot_mask(transformed_mask)

                image_id = test_elem['image_id']

                torch.save(transformed_image, new_dataset_dir + '/test-sat/' + str(image_id) + '_sat.pt')
                torch.save(one_hot_transformed_mask, new_dataset_dir + '/test-mask/' + str(image_id) + '_mask.pt')

            print ('Completed reading dataset in as tensors.')
            print ('*******************************************************************')

    def one_hot_mask(self, mask):
        # road = (mask[0] > 1 / 2).to(torch.long)
        # background = (road == 0)
        # mask = torch.zeros(2, road.shape[0], road.shape[1])
        # mask[0] = road
        # mask[1] = background.to(torch.long)
        # print ('torch.sum(mask[0] == 1): ', torch.sum(mask[0] == 1))
        # print ('torch.sum(mask[0] == 0): ', torch.sum(mask[0] == 0))

        # print ('torch.sum(mask[1] == 1): ', torch.sum(mask[1] == 1))
        # print ('torch.sum(mask[1] == 0): ', torch.sum(mask[1] == 0))

        mask = (mask[0] > 0.5).to(torch.int64)
        # print ('mask: ', mask)
        # plt.imshow(mask.view(mask.shape[0], mask.shape[1], 1))
        # plt.show()

        return mask


class RandomChoice(torch.nn.Module):
    '''
    Randomly choose from multiple transforms to apply the same to multiple images.
    Always apply the last transform.
    '''
    def __init__(self, transforms):
       super().__init__()
       self.transforms = transforms
    #    print ('self.transforms: ', self.transforms)

    def __call__(self, imgs):
        '''A slighty jank way to randomize some transforms and ensure the latter two happen.
        The random transforms are applied equally to all imgs.
        '''
        t = np.random.binomial(n=1, p=0.5, size=len(self.transforms) - 2)
        t = np.append(t, 1) # always do the last two transforms (ToTensor and Normalize)
        t = np.append(t, 1)
        # print ('t: ', t)
        transformed_imgs = []
        for img in imgs:
            for i in range(len(t)):
                # print ('i: ', i)
                if t[i]:
                    # print ('applying')
                    # print ('img: ', img)
                    img = self.transforms[i](img)
            transformed_imgs.append(img)
        return transformed_imgs



augmentation = RandomChoice([
    transforms.RandomVerticalFlip(p=1),
    transforms.RandomHorizontalFlip(p=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),    
])

dataset = RoadsDataset(root_dir='../data/deepglobe-dataset', augmentation=augmentation)
