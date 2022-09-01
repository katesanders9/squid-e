import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torchmetrics
from torchmetrics import Accuracy
# https://github.com/allenai/swig
from JSL.verb.resnet50 import ResNet
from JSL.verb.verbModel import LabelSmoothing
# https://github.com/torrvision/focal_calibration
exec(open("focal_calibration/Losses/focal_loss.py").read())
# https://github.com/tjoo512/belief-matching-framework
exec(open("belief-matching-framework/loss.py").read())

class RNModel(nn.Module):
    def __init__(self, num_classes, conf_index, calibration):
        super(RNModel, self).__init__()

        self.feature_extractor = ResNet()
        state_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet50-19c8e357.pth', model_dir='.')
        self.feature_extractor.load_state_dict(state_dict, strict=False)
        
        self.linear = nn.Linear(2048,num_classes)
        self.calibration = calibration
        self.conf_index = conf_index
        
        if calibration == 'label_smoothing':
            self.loss_function = LabelSmoothing(0.2)
        elif calibration == 'belief_matching':
            self.loss_function = BeliefMatching(0.003)
        elif calibration == 'focal_loss':
            self.loss_function = FocalLoss(3.0)
        elif calibration == 'relaxed_softmax':
            self.loss_function = LabelSmoothing(0.2)
            self.linear = nn.Linear(2048,num_classes+1)
        else:
            self.loss_function = LabelSmoothing(0.0)

        self.mse = nn.MSELoss()
        self.smax = nn.Softmax(dim=1)
        self.accuracy = Accuracy()
        self.ece = torchmetrics.functional.calibration_error
        self.kl = torch.nn.KLDivLoss(reduction="batchmean")

    def forward(self, image, y=None, epoch=0, loss=0):

        pred = self.feature_extractor(image, 6)
        pred = self.linear(pred)   

        if self.calibration == 'relaxed_softmax':
            pred, alpha = torch.index_select(pred, 1, torch.Tensor([0,1]).long().cuda()), torch.index_select(pred, 1, torch.Tensor([2]).long().cuda()) 
            pred = torch.mul(pred, alpha)  

        out = torch.argmax(pred, dim=1)
        out = torch.unsqueeze(out, 1)
        confidence = self.smax(pred).index_select(1, torch.Tensor([self.conf_index]).long().cuda())
        
        if y is None:
            return out
        elif loss == 0:
            if self.calibration in ['label_smoothing', 'belief_matching', 'relaxed_softmax']:
                y = y.squeeze(1)
            return self.loss_function(pred, y)
        elif loss == 1:
            return self.accuracy(out, y)
        else:
            loss = self.mse(confidence, y)
            return loss
        elif l == 3:
            loss = self.ece(confidence, y, n_bins=5)
            return loss
        elif l == 4:
            loss = self.kl(kl_confidence, kl_y)
            return loss
