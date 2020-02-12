import torch 
import torch.nn.functional as F
import torch.nn as nn

class ContBatchNorm3d(nn.modules.batchnorm._BatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'
                             .format(input.dim()))
        # super(ContBatchNorm3d, self)._check_input_dim(input)

    def forward(self, input):
        self._check_input_dim(input)
        return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

class Encorder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(Encorder, self).__init__()
        self.down_conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn1 = ContBatchNorm3d(out_channels)
        #self.bn1 = BatchNorm3d()
        self.relu1 = nn.ReLU()
    def forward(self, x):
        return self.relu1(self.bn1(self.down_conv(x)))

class Decorder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(Decorder, self).__init__()
        self.up_conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn1 = ContBatchNorm3d(out_channels)
        #self.bn1 = BatchNorm3d()
        self.relu1 = nn.ReLU()
    def forward(self, x):
        return self.relu1(self.bn1(self.up_conv(x)))

class Output(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(Output, self).__init__()
        self.up_conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
    def forward(self, x):
        return self.up_conv(x)

class UNet3D(nn.Module):
    def __init__(self, in_channels, n_classes):
        super(UNet3D, self).__init__()
        self.in_channel = in_channels
        self.n_classes = n_classes

        self.ec0 = Encorder(self.in_channel, 32)
        self.ec1 = Encorder(32, 64) 
        self.ec2 = Encorder(64, 64)
        self.ec3 = Encorder(64, 128)
        self.ec4 = Encorder(128, 128)
        self.ec5 = Encorder(128, 256)
        self.ec6 = Encorder(256, 256)
        self.ec7 = Encorder(256, 512)

        self.pool0 = nn.MaxPool3d(2, padding=0)
        self.pool1 = nn.MaxPool3d(2, padding=0)
        self.pool2 = nn.MaxPool3d(2, padding=0)

        self.dc9 = Decorder(512, 512, kernel_size=2, stride=2, padding=0) 
        self.dc8 = Decorder(256+512, 256) 
        self.dc7 = Decorder(256, 256)
        self.dc6 = Decorder(256, 256, kernel_size=2, stride=2, padding=0)
        self.dc5 = Decorder(128+256, 128)
        self.dc4 = Decorder(128, 128)
        self.dc3 = Decorder(128, 128, kernel_size=2, stride=2, padding=0)
        self.dc2 = Decorder(64+128, 64)
        self.dc1 = Decorder(64, 64)
        self.dc0 = Output(64 , n_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        syn0 = self.ec1(self.ec0(x))
        syn1 = self.ec3(self.ec2(self.pool0(syn0)))

        syn2 = self.ec5(self.ec4(self.pool1(syn1)))

        out = self.ec7(self.ec6(self.pool2(syn2)))

        out = torch.cat((self.dc9(out), syn2), dim=1)

        out = self.dc8(out)
        out = self.dc7(out)

        out = torch.cat((self.dc6(out), syn1), dim=1)

        out = self.dc5(out)
        out = self.dc4(out)

        out = torch.cat((self.dc3(out), syn0), dim=1)

        out = self.dc2(out)
        out = self.dc1(out)
        out = self.dc0(out)

        return out 
