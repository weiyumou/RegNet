import torch.nn as nn


class SimpleConv(nn.Module):
    def __init__(self):
        super(SimpleConv, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3, 3),
                      stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3),
                      stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3),
                      stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3),
                      stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )
        self.fc = nn.Linear(in_features=128 * 7 * 7, out_features=10)
        self._initialize_weights()

    def forward(self, x):
        outputs = dict()
        for name, layer in self.named_children():
            if "fc" in name:
                x = x.reshape(x.size(0), -1)
            x = layer(x)
            outputs[name] = x

        for name in outputs:
            outputs[name].retain_grad()

        return outputs

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
            nn.Linear(in_features=28 * 28, out_features=512),
            nn.ReLU(),
            # nn.Dropout()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(in_features=512, out_features=512),
            nn.ReLU(),
            # nn.Dropout()
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
