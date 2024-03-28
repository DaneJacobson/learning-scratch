"""Implementation of VGGNet"""

from PIL import Image

import torch.nn as nn
from torchvision import datasets, transforms, models

class VGGNet_A(nn.Module):

    def __init__(self):
        super(VGGNet_A, self).__init__()

        self.c1 = nn.Sequential([
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ])
        self.c2 = nn.Sequential([
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ])
        self.c3 = nn.Sequential([
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ])
        self.c4 = nn.Sequential([
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ])
        self.c5 = nn.Sequential([
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ])

        self.fc1 = nn.Sequential([
            nn.Linear(in_features=512 * 7 * 7, out_features=4096),
            nn.ReLU(),
        ])
        self.fc2 = nn.Sequential([
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
        ])
        self.fc3 = nn.Sequential([
            nn.Linear(in_features=4096, out_features=1000),
            nn.ReLU(),
        ])

        self.sm = nn.Softmax(dim=1000)

        self.net = nn.Sequential([
            self.c1,
            self.c2,
            self.c3,
            self.c4,
            self.c5,
            self.fc1,
            self.fc2,
            self.fc3,
            self.sm,
        ])

    def forward(self, x):
        return self.net(x)
    

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    # transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

training_set = datasets.ImageNet(root="./data", split="train", transform=transform)

training_set[0].show()