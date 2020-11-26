import torch
import torch.nn as nn
from torchvision import models
import torch.utils.model_zoo


class VGG16_modified(nn.Module):
    def __init__(self, vgg16, n_class=1000):
        super(VGG16_modified, self).__init__()
        self.features = vgg16.features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout()
        )
        # fc_e this is the extra layer
        self.fc_e = nn.Linear(4096*2, n_class)
        self.copy_params_from_vgg(vgg16)

    def sequence(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def forward(self, x1, x2):
        # branch 1
        h1 = self.sequence(x1)
        # branch 2
        h2 = self.sequence(x2)
        # merge two branches
        group = torch.cat((h1, h2), 1)
        o = self.fc_e(group)
        return o

    def copy_params_from_vgg(self, vgg16):
        # skip the last extra layer fc_e layer in vgg16.classifier
        for l1, l2 in zip(vgg16.features, self.features):
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                assert l1.weight.size() == l2.weight.size()
                assert l1.bias.size() == l2.bias.size()
                l2.weight.data = l1.weight.data
                l2.bias.data = l1.bias.data

        for l1, l2 in zip(vgg16.classifier, self.classifier):
            if isinstance(l1, nn.Linear) and isinstance(l2, nn.Linear):
                assert l1.weight.size() == l2.weight.size()
                assert l1.bias.size() == l2.bias.size()
                l2.weight.data = l1.weight.data
                l2.bias.data = l1.bias.data


def main(paras):
    if paras == 'vgg16':
        model = models.vgg16(pretrained=True)
        net_modified = VGG16_modified(model, 3)

    return net_modified


if __name__ == "__main__":
    main()
