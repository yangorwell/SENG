import torch
import torch.nn as nn

def _generate_class(superclass):
    class LayerEx(superclass):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.last_input = None
            self.last_output = None

        def forward(self, x):
            self.last_input = x.data
            self.last_output = super().forward(x)
            if self.training:
                self.last_output.retain_grad()
            return self.last_output
    LayerEx.__name__ = superclass.__name__ + 'Ex'
    return LayerEx

Conv2dEx = _generate_class(nn.Conv2d)
LinearEx = _generate_class(nn.Linear)
BatchNorm2dEx = _generate_class(nn.BatchNorm2d)
