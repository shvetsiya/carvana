from torch import nn, log


class Loss:
    def __init__(self, dice_weight=1):
        self.nll_loss = nn.BCELoss()
        self.dice_weight = dice_weight

    def __call__(self, preds, targets):
        loss = self.nll_loss(preds, targets)
        if self.dice_weight:
            smooth = 1e-10
            num = targets.size(0)# batch size
            
            m1 = preds.view(num, -1)                
            m2 = targets.view(num, -1)

            intersaction = (m1*m2).sum(1) + smooth
            tsquares = m1.sum(1) + m2.sum(1) + smooth            
            # dice ceoff per batch
            dice = 2*(intersaction/tsquares).mean()
            loss += 1.0 - dice
	    #loss -= log(dice) #ternaus	
        return loss
