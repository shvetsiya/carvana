import cv2
import torch
import numpy as np
from tensorboardX import SummaryWriter
import shutil

class Callback:
    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class TensorBoardVisualizerCallback(Callback):
    def __init__(self, path_to_files):
        """
        Callback is executed every training epoch. The goal is to display 
        the result of the last validation batch in Tensorboard
        Args:
            path_to_files (str): The path where to store the log files
        """
        self.path_to_files = path_to_files

    def mask_overlay(self, image, mask, color=(0, 255, 0)):
        mask = np.dstack((mask, mask, mask)) * np.array(color)
        mask = mask.astype(np.uint8)
        return cv2.addWeighted(mask, 0.5, image, 0.5, 0.)  # image * α + mask * β + λ

    def representation(self, image, mask):
        """
         Given a mask and an image this method returns
         one image representing 3 patches of the same image.
         These patches represent:
            - The original image
            - The predicted/original mask
            - The mask applied to the image
        Args:
            image (np.ndarray): The original image
            mask (np.ndarray): The predicted/original mask

        Returns (np.ndarray):
            An image of size (original_image_height, (original_image_width * 3))
            showing 3 patches of the original image
        """

        H, W, C = image.shape
        results = np.zeros((H, 3*W, 3), np.uint8)
        blue_mask = np.zeros((H*W, 3), np.uint8)
        
        pb = np.where(mask.flatten()==1)[0]
        blue_mask[pb] = np.array([0, 0, 255])
        blue_mask = blue_mask.reshape(H, W, 3)

        overlay_imgs = self.mask_overlay(image, mask)
        
        results[:, 0: W] = image
        results[:, W: 2*W] = blue_mask
        results[:, 2*W: 3*W] = overlay_imgs
        return results

    def __call__(self, *args, **kwargs):

        epoch = kwargs['epoch']
        last_images, last_targets, last_preds = kwargs['last_valid_batch']

        writer = SummaryWriter(self.path_to_files)
        for i, (image, target_mask, pred_mask) in enumerate(zip(last_images, last_targets, last_preds)):

            image = (255*image.data).cpu().numpy().astype(np.uint8)
            image = np.transpose(image, (1, 2, 0))  # Invert c, h, w to h, w, c

            target_mask = (target_mask.data).cpu().numpy().astype(np.uint8).squeeze()
            pred_mask = (pred_mask.data).cpu().numpy().astype(np.uint8).squeeze()

            if image.shape[0] > 512:  # We don't want images on tensorboard to be too large
                image = cv2.resize(image, (512, 512))
                target_mask = cv2.resize(target_mask, (512, 512))
                pred_mask = cv2.resize(pred_mask, (512, 512))

            expected_result = self.representation(image, target_mask)
            pred_result = self.representation(image, pred_mask)
            writer.add_image("Epoch_" + str(epoch) + '-Image_' + str(i + 1) + '-Expected', expected_result, epoch)
            writer.add_image("Epoch_" + str(epoch) + '-Image_' + str(i + 1) + '-Predicted', pred_result, epoch)
            if i == 1:  # 2 Images are sufficient
                break
        writer.close()


class TensorBoardLoggerCallback(Callback):
    def __init__(self, path_to_files):
        """
        Callback intended to be executed at each epoch
        of the training which goal is to add valuable
        information to the tensorboard logs such as the losses
        and accuracies
        Args:
            path_to_files (str): The path where to store the log files
        """
        self.path_to_files = path_to_files

    def __call__(self, *args, **kwargs):
        epoch = kwargs['epoch']

        writer = SummaryWriter(self.path_to_files)
        writer.add_scalar('data/train_loss', kwargs['train_loss'], epoch)
        writer.add_scalar('data/train_dice', kwargs['train_dice'], epoch)
        writer.add_scalar('data/valid_loss', kwargs['valid_loss'], epoch)
        writer.add_scalar('data/valid_dice', kwargs['valid_dice'], epoch)
        writer.close()


class ModelSaverCallback(Callback):
    def __init__(self, path_to_model, path_to_best_model):
        """
        Callback intended to be executed each time a whole train pass
        get finished. This callback saves the model in the given path
        Args:
            best_valid_loss (double): serve to identify the best model 
            path_to_model (str): The path where to store the model
            path_to_best_model (str): The path where the best model is stored
        """
        self.best_valid_loss = float('inf')
        self.path_to_model = path_to_model
        self.path_to_best_model = path_to_best_model
        
    def __call__(self, *args, **kwargs):        
        net = kwargs['net']
        epoch = kwargs['epoch']
        valid_loss = kwargs['valid_loss']
        
        is_best = valid_loss < self.best_valid_loss
        self.best_valid_loss = min(self.best_valid_loss, valid_loss)
        
        state = {"epoch": epoch+1, "state_dict": net.state_dict(), "valid_loss": valid_loss}
        torch.save(state, self.path_to_model)
        if is_best:
            shutil.copyfile(self.path_to_model, self.path_to_best_model)


class SimpleLoggerCallback(Callback):
    def __init__(self, log_file):
        """
        Callback intended to be executed each time a whole train pass
        get finished. This callback saves metrics in logfile
        Args:
            file_name (str): The path where to store the metrics
        """
        self.filename = log_file

    def __call__(self, *args, **kwargs):        
        epoch = kwargs['epoch']
        train_loss = kwargs['train_loss']
        valid_loss = kwargs['valid_loss']
        train_dice = kwargs['train_dice']
        valid_dice = kwargs['valid_dice']
                          
        log_string =  "epoch = {},\t".format(epoch)
        log_string += "train_loss = {:03f},\t".format(train_loss)
        log_string += "train_dice = {:03f},\t".format(train_dice)
        log_string += "valid_loss = {:03f},\t".format(valid_loss)
        log_string += "valid_dice = {:03f}".format(valid_dice)

        with open(self.filename, 'a') as f:
            f.write(log_string)
            f.write('\n')            
