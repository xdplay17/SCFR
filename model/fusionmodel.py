import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from utils.tool import load_preweights

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        output = self.conv(result)
        output = self.sigmoid(output)
        return output

class FusionModel(nn.Module):
    def __init__(self, hash_bit):
        super(FusionModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu5 = nn.ReLU(inplace=True)
        self.localbranch = SpatialAttention(kernel_size=13)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 256),
        )
        self.fc = nn.Linear(512, hash_bit)
        self.tanh = nn.Tanh()


    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.conv5(x)
        x = self.relu5(x)
        attention = self.localbranch(x)
        local_fea = x * attention

        x = self.maxpool3(x)
        x = x.view(x.size(0), 256 * 6 * 6)

        glo_fea = self.classifier(x)
        glo_fea = glo_fea.unsqueeze(-1).unsqueeze(-1)
        fea = torch.cat([glo_fea.expand(local_fea.size()), local_fea], dim=1)
        fea = self.avgpool(fea)
        fea = fea.view(fea.size(0), -1)
        fea = self.fc(fea)
        fea = self.tanh(fea)
        fea = F.normalize(fea, dim=1)
        return fea


if __name__ == '__main__':
    net = FusionModel(hash_bit=64)
    preweight = 'alexnet-owt-7be5be79.pth'
    weight = load_preweights(net, preweights=preweight)
    net.load_state_dict(weight)

    a = torch.randn(3, 3, 224, 224)
    x = net(a)
    print(x.shape)
