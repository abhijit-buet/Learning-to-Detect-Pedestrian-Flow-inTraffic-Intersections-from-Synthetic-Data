import torch
import torch.nn as nn
from torchvision import models
import torch.utils.model_zoo


class AlexNet_modified(nn.Module):
    def __init__(self, alexnet, n_class=1000):
        super(AlexNet_modified, self).__init__()
        # conv1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
        # self.bn1 = nn.BatchNorm2d(64, momentum=0.1)
        self.relu1 = nn.ReLU(inplace=True)
        # LRN(local_size=5, alpha=0.0001, beta=0.75)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        # conv2
        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        # self.bn2 = nn.BatchNorm2d(192, momentum=0.1)
        self.relu2 = nn.ReLU(inplace=True)
        # LRN(local_size=5, alpha=0.0001, beta=0.75)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        # conv3
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        # self.bn3 = nn.BatchNorm2d(384, momentum=0.1)
        self.relu3 = nn.ReLU(inplace=True)

        # conv4
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        # self.bn4 = nn.BatchNorm2d(256, momentum=0.1)
        self.relu4 = nn.ReLU(inplace=True)

        # conv5
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        # self.bn5 = nn.BatchNorm2d(256, momentum=0.1)
        self.relu5 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=2)
        # self.pool5 = nn.AdaptiveAvgPool2d(6)
        self.drop5 = nn.Dropout()

        # fc6
        self.fc6 = nn.Linear(256 * 6 * 6, 4096)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout()

        # fc7
        self.fc7 = nn.Linear(4096, 4096)
        self.relu7 = nn.ReLU(inplace=True)

        # fc8 this is the extra layer
        self.fc8 = nn.Linear(4096*2, n_class)
        self.copy_params_from_alexnet(alexnet)

    def sequence(self, x):
        h = x
        # h = self.relu1(self.bn1(self.conv1(h)))
        h = self.relu1(self.conv1(h))
        h = self.pool1(h)

        # h = self.relu2(self.bn2(self.conv2(h)))
        h = self.relu2(self.conv2(h))
        h = self.pool2(h)

        # h = self.relu3(self.bn3(self.conv3(h)))
        h = self.relu3(self.conv3(h))

        # h = self.relu4(self.bn4(self.conv4(h)))
        h = self.relu4(self.conv4(h))

        # h = self.relu5(self.bn5(self.conv5(h)))
        h = self.relu5(self.conv5(h))
        h = self.pool5(h)
        h = self.drop5(h)

        h = h.view(-1, 256 * 6 * 6)
        h = self.relu6(self.fc6(h))
        h = self.drop6(h)

        h = self.relu7(self.fc7(h))

        return h

    def forward(self, x1, x2):
        # branch 1
        h1 = self.sequence(x1)

        # branch 2
        h2 = self.sequence(x2)

        # merge two branches

        group = torch.cat((h1, h2), 1)
        o = self.fc8(group)
        return o

    def copy_params_from_alexnet(self, alexnet):
        features = [
            self.conv1, self.relu1,
            self.pool1,
            self.conv2, self.relu2,
            self.pool2,
            self.conv3, self.relu3,
            self.conv4, self.relu4,
            self.conv5, self.relu5,
            self.pool5,
        ]
        for l1, l2 in zip(alexnet.features, features):
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                assert l1.weight.size() == l2.weight.size()
                assert l1.bias.size() == l2.bias.size()
                l2.weight.data = l1.weight.data
                l2.bias.data = l1.bias.data
        for i, name in zip([1, 4], ['fc6', 'fc7']):
            l1 = alexnet.classifier[i]
            l2 = getattr(self, name)
            l2.weight.data = l1.weight.data.view(l2.weight.size())
            l2.bias.data = l1.bias.data.view(l2.bias.size())


def main(paras):
    if paras == 'alexnet':
        model = models.alexnet(pretrained=True)
        net_modified = AlexNet_modified(model, 3)

    return net_modified


if __name__ == "__main__":
    main()
