import math
import inspect
from dataclasses import dataclass
from sophia import SophiaG

import torch
import torch.nn as nn
from torch.nn import functional as F

optimizer_dict = {'adamw': torch.optim.AdamW,
                  'sophiag': SophiaG
                  }





class MLP(nn.Module):

    def __init__(self, input, output, bias=False):
        super().__init__()
        self.c_fc = nn.Linear(input, output, bias=bias)
        self.loss = torch.nn.CrossEntropyLoss()


    def forward(self, x, targets=None):
        logits = self.c_fc(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            if not isinstance(targets, int):

                loss = self.loss(logits, targets)
            else:
                loss = None
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            loss = None

        return logits, loss




        return x


