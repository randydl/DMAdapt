import torch
import torch.nn as nn


__all__ = ['LeNet']


class LeNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(LeNet, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, 6, 5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(400, 120),
            nn.ReLU(inplace=True)
        )

        self.classifier = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, num_classes)
        )

    def forward(self, sX, tX=None):
        sF = self.feature_extractor(sX)
        sC = self.classifier(sF)
        if tX is None: return sC
        tF = self.feature_extractor(tX)
        return sC, sF, tF


if __name__ == '__main__':
    model = LeNet(1, 10)
    x = torch.randn(2, 1, 32, 32)
    preds = model(x, x.clone())
