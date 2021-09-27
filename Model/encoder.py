import torch
import torch.nn as nn
from torch.utils import model_zoo
from torchvision.models import resnet

from Model.resnet import ResNetFeatModule
from Model.resnet import BasicBlock


class encoder(nn.Module):
    def __init__(self, flags):
        super(encoder, self).__init__()
        self.feature_extractor = ResNetFeatModule(BasicBlock, [2, 2, 2, 2])
        self.load_resnet(flags.state_dict,self.feature_extractor)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, flags.embed_size)
    def forward(self,x):
        x = self.feature_extractor(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        # embed = Variable(embed.data)
        # embed = embed.view(embed.size(0), -1)
        # embed = self.linear(embed)
        return x

    def load_resnet(self, state_dict, network):
        try:
            tmp = torch.load(state_dict)
            if 'state' in tmp.keys():
                pretrained_dict = tmp['state']
            else:
                pretrained_dict = tmp
        except:
            pretrained_dict = model_zoo.load_url(state_dict)

        model_dict = network.state_dict()

        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                               k in model_dict and v.size() == model_dict[k].size()}

        print('model dict keys:', len(model_dict.keys()), 'pretrained keys:', len(pretrained_dict.keys()))
        print('model dict keys:', model_dict.keys(), 'pretrained keys:', pretrained_dict.keys())
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        network.load_state_dict(model_dict)

    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.

        :param fine_tune: Allow?
        """
        for p in self.resnet.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune

