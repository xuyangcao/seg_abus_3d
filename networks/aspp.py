import torch
import torch.nn as nn

class _ASPPModule(nn.Module):
    '''
    basic ASPP module
    '''
    def __init__(self, inplanes, planes, kernel_size, padding, dilation):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv3d(inplanes, planes, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation)
        self.gn = nn.GroupNorm(16, planes)
        self.relu = nn.PReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.relu(self.gn(x))

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP(nn.Module):
    '''
    ASPP module for capturing multi-scale feature
    '''
    def __init__(self, inplanes, planes=64, dilations=[1, 6, 12, 18], drop_rate=0.5):
        super(ASPP, self).__init__()
        self.aspp1 = _ASPPModule(inplanes, planes, 1, padding=0, dilation=dilations[0])
        self.aspp2 = _ASPPModule(inplanes, planes, 3, padding=dilations[1], dilation=dilations[1])
        self.aspp3 = _ASPPModule(inplanes, planes, 3, padding=dilations[2], dilation=dilations[2])
        self.aspp4 = _ASPPModule(inplanes, planes, 3, padding=dilations[3], dilation=dilations[3])
        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)),
                                             nn.Conv3d(inplanes, planes, 1, stride=1, bias=False),
                                             nn.BatchNorm3d(planes),
                                             nn.PReLU())
        self.conv1 = nn.Conv3d(5*planes, planes, bias=False)
        self.gn1 = nn.GroupNorm(16, planes)
        self.relu = nn.PReLU()
        self.dropout = nn.Dropout3d(drop_rate)

        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.gn1(x)
        x = self.relu(x)
        x = self.dropout(x)

        return x 

    def _init_weight():
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
