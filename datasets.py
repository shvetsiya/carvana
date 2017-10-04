import cv2
import numpy as np
import pandas as pd
import random
import torch
from torch.utils.data import Dataset

from image_transformations import *
from config import *


class CarvanaTrainDataset(Dataset):
    def __init__(self, ids):
        self.ids = ids

    def __len__(self):
        return len(self.ids)

    def resize(self, img):
        return cv2.resize(img, (IM_SIZE_W, IM_SIZE_H))

    def crop(self, nimg, nmask):
        crop_x = random.randint(0, ORIGINAL_HEIGHT - IM_SIZE_H)
        crop_y = random.randint(0, ORIGINAL_WIDTH - IM_SIZE_W)
        cr_img = nimg[crop_x:crop_x + IM_SIZE_H, crop_y:crop_y + IM_SIZE_W]
        cr_mask = nmask[crop_x:crop_x + IM_SIZE_H, crop_y:crop_y + IM_SIZE_W]
        return cr_img, cr_mask

    def make_whole(self, nimg):
        """ Make 1920x1280
        """
        new_img = np.zeros((1280, 1920))
        new_img[:, 1:ORIGINAL_WIDTH+1] = nimg
        new_img[:, 0] = nimg[:, 0]
        new_img[:, -1] = nimg[:, -1]
        return new_img

    def __getitem__(self, index):
        id = self.ids[index]
        img = cv2.imread('{}/{}.jpg'.format(TRAIN_DIR, id))
        mask = cv2.imread('{}/{}_mask.png'.format(MASKS_DIR, id), cv2.IMREAD_GRAYSCALE)
        """
        if ORIGINAL_HEIGHT == IM_SIZE_H and ORIGINAL_WIDTH == IM_SIZE_W:
            img = self.make_whole(img)
            mask = self.make_whole(mask)
        else:    
            #img, mask = self.crop(img, mask)
        """    
        img = self.resize(img)
        mask = self.resize(mask)        
        
        
        img = random_hue_saturation_value(img,
                                          hue_shift_limit=(-50, 50),
                                          sat_shift_limit=(-5, 5),
                                          val_shift_limit=(-15, 15)
        )
        
        #  img, mask = random_shift_scale_rotate(
        #  img, mask, shift_limit=(-0.0625, 0.0625), scale_limit=(-0.1, 0.1), rotate_limit=(-0, 0))
        img, mask = random_horizontal_flip(img, mask)

        mask = np.expand_dims(mask, axis=2)
        img = np.array(img, np.float32) / 255
        mask = np.array(mask, np.float32) / 255
        
        return torch.from_numpy(img).permute(2, 0, 1), torch.from_numpy(mask).permute(2, 0, 1)


class CarvanaTestDataset(Dataset):
    def __init__(self, ids):
        self.ids = ids
        print(ids[:10])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        img_id = self.ids[index]
        img = cv2.imread('{}/{}.jpg'.format(TEST_DIR, img_id))
        img = cv2.resize(img, (IM_SIZE_W, IM_SIZE_H))
        img = np.array(img, np.float32) / 255
        return torch.from_numpy(img).permute(2, 0, 1), img_id
