import torch.nn as nn
import torch
from module.modules import augment



class ensenble(nn.Module):
    def __init__(self,model1, model2, num_labels):
        super(ensenble, self).__init__()
        self.model1= model1
        self.model2 = model2
        self.inplanes=512
        self.inplanes=1024
        self.gvp1 = nn.AdaptiveAvgPool2d((1, 1))
        self.gvp = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(1024, 1024, bias=False)
        self.drop1 = nn.Dropout(0.3)

        self.act1 = nn.ReLU(inplace=True)
        self.drop2 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(1024, num_labels, bias=False)

        self.augment=augment()
        self.trainable_parameters=[]
        for name, param in self.named_parameters():
            if ('conv' in name) and ('weight' in name):
                self.trainable_parameters.append(param)
            else:
                self.trainable_parameters.append(param)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                #nn.init.constant_(m.bias, 0.2)


    def _make_layer(self, block, num_filters, groups=1):
        layers = []
        for planes in num_filters:
            layers.append(block(self.inplanes, planes, groups=groups))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, input, prob):
        im, im_c = input
        im = self.augment(im, prob)

        out1 = self.model1(im, 1)
        out2 = self.model2(im_c,1)
        out = torch.cat([out1, out2], dim=1)
        out = out.view(out.size(0), -1)
        out = self.drop1(out)
        out = self.fc1(out)
        out = self.act1(out)
        out = self.drop2(out)
        out =self.fc2(out)
        return out

