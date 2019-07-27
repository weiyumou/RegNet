import torch.nn as nn
import torch.nn.functional as F


class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.size(0), -1)


class SimpleConv(nn.Module):
    def __init__(self):
        super(SimpleConv, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3),
                      stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3),
                      stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3),
                      stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3),
                      stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )
        self.fc = nn.Sequential(
            Flatten(),
            nn.Linear(in_features=128 * 8 * 8, out_features=10)
        )

        self._initialize_weights()

    def forward(self, x):
        outputs = dict()
        last_name = None
        for name, layer in self.named_children():
            x = layer(x)
            outputs[name] = x
            last_name = name
        return outputs if self.training else outputs[last_name]

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Sequential(
            Flatten(),
            nn.Linear(in_features=28 * 28, out_features=512),
            nn.ReLU(),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(in_features=512, out_features=512),
            nn.ReLU(),
        )

        self.fc3 = nn.Linear(in_features=512, out_features=10)

    def forward(self, x):
        outputs = dict()
        last_name = None
        for name, layer in self.named_children():
            x = layer(x)
            outputs[name] = x
            last_name = name
        return outputs if self.training else outputs[last_name]


class VGGConvBlock(nn.Module):

    def __init__(self, num_in_channels, num_out_channels):
        super(VGGConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=num_in_channels, out_channels=num_out_channels,
                              kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(num_features=num_out_channels)

    def forward(self, x):
        for _, layer in self.named_children():
            x = layer(x)
        return F.relu(x)


class VGGMaxConvBlock(VGGConvBlock):

    def __init__(self, num_in_channels, num_out_channels):
        super().__init__(num_in_channels, num_out_channels)
        self.maxpool = nn.MaxPool2d(kernel_size=2)


class VGG(nn.Module):

    def __init__(self, num_channels, num_classes):
        super(VGG, self).__init__()
        num_in_channels, num_out_channels = num_channels, 64
        num_conv_layers = [2, 2, 3, 3, 3]

        count = 0
        for num_conv in num_conv_layers:
            self.add_module(f"conv{count + 1}", VGGConvBlock(num_in_channels, num_out_channels))
            count += 1
            for _ in range(1, num_conv - 1):
                self.add_module(f"conv{count + 1}", VGGConvBlock(num_out_channels, num_out_channels))
                count += 1

            self.add_module(f"conv{count + 1}", VGGMaxConvBlock(num_out_channels, num_out_channels))
            count += 1
            num_in_channels, num_out_channels = num_out_channels, min(512, num_out_channels * 2)

        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
            nn.Linear(in_features=512, out_features=num_classes)
        )
        self._initialize_weights()

    def forward(self, x):
        outputs = dict()
        last_name = None
        for name, layer in self.named_children():
            x = layer(x)
            outputs[name] = x
            last_name = name
        return outputs if self.training else outputs[last_name]

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
