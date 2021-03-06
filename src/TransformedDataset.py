import torch
import os

# inspiration taken from https://medium.com/codex/saving-and-loading-transformed-image-tensors-in-pytorch-f37b4daa9658

class TransformedDataset(torch.utils.data.Dataset):
    def __init__(self, img, mask, has_labels=True, return_name=False, small_set=False):
        self.img = img  #img path
        self.mask = mask  #mask path
        self.len = 0
        self.has_labels = has_labels
        self.return_name = return_name
        with os.scandir(self.img) as it:
            for entry in it:
                if not entry.name.startswith('.'): # don't count hidden files
                    self.len += 1
                    if small_set and self.len == 2: # small len
                        break


        # self.len = len(os.listdir(self.img)) # one more than actually there? It might be picking up .DS_Store

    def __getitem__(self, index):
        # ls_img = []
        # with os.scandir(self.img) as it:
        #     cur_index = 0
        #     for entry in it:
        #         if not entry.name.startswith('.'): # avoid hidden files
        #             ls_img.append(entry)
        #             cur_index += 1
        #             if cur_index > index: # we only need to count up to index
        #                 break
        # ls_img = sorted(ls_img)

        ls_img = sorted(os.listdir(self.img))
        # print ('ls_img: ', ls_img)
        toRemove = []
        for elem in ls_img:
            if elem[0] == '.':
                toRemove.append(elem)
        for elem in toRemove:
            ls_img.remove(elem)
            


        if (self.has_labels):
            ls_mask = sorted(os.listdir(self.mask))
            toRemove = []
            for elem in ls_mask:
                if elem[0] == '.':
                    toRemove.append(elem)
            for elem in toRemove:
                ls_mask.remove(elem)



            # ls_mask = []
            # with os.scandir(self.mask) as it:
            #     cur_index = 0
            #     for entry in it:
            #         if not entry.name.startswith('.'): # avoid hidden files
            #             ls_mask.append(entry)
            #             cur_index += 1
            #             if cur_index > index: # we only need to count up to index
            #                 break

        img_file_path = os.path.join(self.img, ls_img[index])
        # print ('img_file_path: ', img_file_path)
        img_tensor = torch.load(img_file_path)

        if (self.has_labels):
            mask_file_path = os.path.join(self.mask, ls_mask[index])
            mask_tensor = torch.load(mask_file_path)


        toReturn = [img_tensor]
        if self.has_labels:
            toReturn.append(mask_tensor)
        if self.return_name:
            toReturn.append(ls_img[index])
        return tuple(toReturn)

        
        # if (self.has_labels):
        #     return img_tensor, mask_tensor
        # else:
        #     return img_tensor, ls_img[index]

    def __len__(self):
        return self.len


# train_labeled_dataset = TransformedDataset('../data/deepglobe-dataset-pt/train-labeled-sat', '../data/deepglobe-dataset-pt/train-labeled-mask')
# print ('train_labeled_dataset.__len__(): ', train_labeled_dataset.__len__())
# for i in range(1, train_labeled_dataset.__len__()):
#     print (train_labeled_dataset[i])
