import numpy as np
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torchmetrics
from torchmetrics import Accuracy
# https://github.com/allenai/swig
from JSL.verb.resnet50 import ResNet
from JSL.verb.verbModel import LabelSmoothing


class MCModel(nn.Module):
    def __init__(self, num_classes, conf_index):
        super(MCModel, self).__init__()

        self.feature_extractor = ResNet()
        state_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet50-19c8e357.pth', model_dir='.')
        self.feature_extractor.load_state_dict(state_dict, strict=False)
        self.conf_index = conf_index 
        self.linear = nn.Linear(2048, num_classes)
        self.dropout = nn.Dropout(0.5)
        self.loss_function = LabelSmoothing(0.0)
        self.mse = nn.MSELoss()
        self.smax = nn.Softmax(dim=1)
        self.accuracy = Accuracy()

    def forward(self, image, y=None, epoch=0, loss=0):

        pred = self.feature_extractor(image, epoch)
        self.dropout.train()
        pred = self.dropout(pred)
        pred = self.linear(pred)
       
        if loss == 0 and not y is None:
            y = y.squeeze(1)
            return self.loss_function(pred, y)
 
        preds = [self.linear(self.dropout(self.feature_extractor(image))) for _ in range(50)]
        pred = torch.stack(preds).mean(axis=0)
        out = torch.argmax(pred, dim=1)
        out = torch.unsqueeze(out, 1)
        confidence = self.smax(pred).index_select(1, torch.Tensor([self.conf_index]).long().cuda())

        if y is None:
            return out
        elif loss == 1:
            return self.accuracy(out, y)
        else:
            return self.mse(confidence, y)