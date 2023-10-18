import torchvision
from torch import nn
from efficientnet_pytorch import EfficientNet

class EfficientNetV2(nn.Module):
    def __init__(self):
        super().__init__()
        # Two output classes: Hot dog, not hot dog
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b0', num_classes=2)

    def forward(self, x):
        x = self.efficientnet(x)
        return x