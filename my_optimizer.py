import numpy as np
import torch
from torch.optim import Optimizer



class WeirdDescent(Optimizer):



    def __init__(self, parameters, lr=1-3):
        defaults = {"lr": lr}

        # Constructor of the parent
        super().__init__(parameters, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        if not self.state["step"]:
            self.state["step"] = 1
        else:
            self.state["step"] += 1


        c = 1

        if self.state["step"] % 100 == 0:
            c = 100
        grad = None

        while grad is None:
            param_group = np.random.choice(self.param_groups)
            tensor = np.random.choice(param_group["params"])
            grad = tensor.grad.data

