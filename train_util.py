from collections import OrderedDict
from tqdm import tqdm
import cv2

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model.losses import Loss
import tools 
from config import *


class CarvanaSegmenationTrain:
    def __init__(self, net:nn.Module, num_epochs, learning_rate, load_model=False):
        """
        The classifier for carvana used for training and launching predictions
        Args:
            net (nn.Module): The neural net module containing the definition of your model
            num_epochs (int): The maximum number of epochs which is used to train the model 
        """
        self.net = nn.DataParallel(net, device_ids=GPU_IDS).cuda() 
        self.model_path_last = str(SAVED_MODEL)
        self.model_path_best = str(BEST_MODEL)
        self.epoch_old = 0
        if load_model:
            self.load_best_model(self.model_path_best)

        self.num_epochs = num_epochs
        self.criterion = Loss()
        self.optimizer = optim.RMSprop(self.net.parameters(), lr=learning_rate)
        self.lr_scheduler = ReduceLROnPlateau(self.optimizer,
                                              mode='min',
                                              factor=0.3,
                                              patience=2,
                                              verbose=True,
                                              min_lr=1e-7)
        #self.optimizer = optim.Adam(model.parameters(), weight_decay=regularization)

    def load_best_model(self, model_path) -> None:
        """
        Restore a model parameters from the one given in argument
        Args:
            model_path=(str): The path to the model to restore

        """
        state = torch.load(model_path)
        self.net.load_state_dict(state['state_dict'])
        self.epoch_old = state['epoch']
        print('Loaded model from epoch {}'.format(self.epoch_old)) # '{epoch}'.format(**state)

    def dice_loss(self, pred, target):
        smooth = 1e-10
        num = target.size(0)# batch size
        m1 = pred.view(num, -1)
        m2 = target.view(num, -1)
        #thanks to broadcasting
        intersection = (m1*m2).sum(1) + smooth
        tsquares = m1.sum(1) + m2.sum(1) + smooth
        
        return 2*(intersection/tsquares).mean()        

    def train_epoch_step(self, epoch:int, train_loader: DataLoader):
        losses = tools.AverageMeter()
        dice_c = tools.AverageMeter()
        # Set train mode.
        self.net.train()

        it_count = len(train_loader)
        batch_size = train_loader.batch_size
        
        with tqdm(total=it_count,
                  desc="Epochs {}/{}".format(epoch+1, self.num_epochs),
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{remaining}{postfix}]'
                  ) as pbar:
            for images, targets in train_loader:
                images = Variable(images.cuda())
                targets = Variable(targets.cuda())

                # Compute output:
                output = self.net(images)
                preds = output.ge(THRESHOLD).float()
                
                #metrics
                loss = self.criterion(output, targets)
                dice = self.dice_loss(preds, targets)

                # Update weights.
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                losses.update(loss.data[0], batch_size)
                dice_c.update(dice.data[0], batch_size)

                # Update pbar
                pbar.set_postfix(OrderedDict(loss='{0:1.5f}'.format(loss.data[0]),
                                             dice_coeff='{0:1.5f}'.format(dice.data[0])))
                pbar.update(1)
                
        return losses.avg, dice_c.avg
                  
    
    def validate(self, valid_loader: DataLoader):
        losses = tools.AverageMeter()
        dice_c = tools.AverageMeter()

        # Set evaluation mode.
        self.net.eval()
        
        it_count = len(valid_loader)
        batch_size = valid_loader.batch_size

        images = None  # To save the last image batch
        targets = None # To save the last target batch
        preds = None   # To save the last prediction batch
        with tqdm(total=it_count, desc="Validating", leave=False) as pbar:
            for images, targets in valid_loader:
                # Volatile is used for pure inference mode
                images = Variable(images.cuda(), volatile=True) 
                targets = Variable(targets.cuda(), volatile=True)

                # Compute output:
                output = self.net(images)
                preds = output.ge(THRESHOLD).float()

                #metrics
                loss = self.criterion(output, targets)
                dice = self.dice_loss(preds, targets)

                losses.update(loss.data[0], batch_size)
                dice_c.update(dice.data[0], batch_size)

                pbar.update(1)
        return losses.avg, dice_c.avg, images, targets, preds


    def train(self, train_loader: DataLoader, valid_loader: DataLoader, callbacks):

        for epoch in range(self.num_epochs):
            train_loss, train_dice = self.train_epoch_step(epoch, train_loader)
            valid_loss, valid_dice, last_images, last_targets, last_preds = self.validate(valid_loader)
            
            # Reduce learning rate if is is needed
            self.lr_scheduler.step(valid_loss, epoch)        

            # save last and the best models
            if callbacks:
                for cb in callbacks:
                    cb(net=self.net,
                    last_valid_batch=(last_images, last_targets, last_preds),
                    epoch=self.epoch_old+epoch,
                    train_loss=train_loss, train_dice=train_dice,
                    valid_loss=valid_loss, valid_dice=valid_dice
                    )
                    
            print("train_loss = {:03f}, train_dice = {:03f}\n"
                  "valid_loss = {:03f}, valid_dice = {:03f}"
                  .format(train_loss, train_dice, valid_loss, valid_dice))

class CarvanaSegmenationTest:
    def __init__(self, net:nn.Module, pred_folder):
        """
        The classifier for carvana used predictions
        Args:
            net (nn.Module): The neural net module containing the definition of your model
            num_epochs (int): The maximum number of epochs which is used to train the model 
        """
        #net = UNet()
        self.net = nn.DataParallel(net, device_ids=GPU_IDS).cuda() 
        self.model_path_best = str(BEST_MODEL)

        self.load_best_model(self.model_path_best)
        self.pred_folder = pred_folder

        if not os.path.exists(self.pred_folder):
            os.makedirs(self.pred_folder)

    def load_best_model(self, model_path) -> None:
        """
        Restore a model parameters from the one given in argument
        Args:
            model_path=(str): The path to the model to restore

        """
        state = torch.load(model_path)
        self.net.load_state_dict(state['state_dict'])
        print('Loaded model from epoch {epoch}'.format(**state)) 
        
    def predict(self, test_loader):
        """
        Launch the prediction on the given loader and pass
        each predictions to the given callbacks.
        Args:
            test_loader (DataLoader): The loader containing the test dataset
            callbacks (list): List of callbacks functions to call at prediction pass
        """
        # Switch to evaluation mode
        self.net.eval()

        it_count = len(test_loader)
        with tqdm(total=it_count, desc="Predict") as pbar:
            for images, file_names in test_loader:
                images = Variable(images.cuda(), volatile=True)
                # forward
                batch_probs = (255*(self.net(images).data)).cpu().numpy().squeeze().astype(np.uint8)
                self.write_batch(batch_probs, file_names)
                
                pbar.update(1)
                
    def write_batch(self, batch_probs, ids):            
        for (pred, name) in zip(batch_probs, ids):
            filename = os.path.join(self.pred_folder, name + '_pred_mask.png')
            pred = cv2.resize(pred, (ORIGINAL_WIDTH, ORIGINAL_HEIGHT))
            cv2.imwrite(filename, pred)
