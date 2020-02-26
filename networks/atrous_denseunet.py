import torch 
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, padding, dilation):
        super(_DenseLayer, self).__init__()
        # modules for bottle neck layer
        self.add_module('norm1', nn.BatchNorm3d(num_input_features))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv1', nn.Conv3d(num_input_features, bn_size * growth_rate, 
                                            kernel_size=1, stride=1, bias=False))
        # modules for dense  layer
        self.add_module('norm2', nn.BatchNorm3d(bn_size * growth_rate))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module('conv2', nn.Conv3d(bn_size * growth_rate, growth_rate, 
                                            kernel_size=3, stride=1, padding=padding, 
                                            dilation=dilation, bias=False))
        self.drop_rate = float(drop_rate)

    def bn_function(self, inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))

        return bottleneck_output

    def forward(self, inputs):
        prev_features = inputs
        bottleneck_output = self.bn_function(prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)

        return new_features


class _DenseBlock(nn.ModuleDict):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, 
                drop_rate, output_stride, use_dilation=False):
        super(_DenseBlock, self).__init__()
        if output_stride == 2:
            dilations = [2, 4, 8, 16]
        elif output_stride == 4:
            dilations = [2, 4, 8, 16]
        else:
            dilations = [2, 2, 4, 4]


        if use_dilation:
            num_dilations = len(dilations)
            # layers wiout atrous convolution
            for i in range(num_layers - num_dilations):
                layer = _DenseLayer(
                        num_input_features + i * growth_rate,
                        growth_rate=growth_rate,
                        bn_size=bn_size,
                        drop_rate=drop_rate,
                        padding=1,
                        dilation=1
                        )
                self.add_module('denselayer%d' % (i + 1), layer)
            # layers with atrous convolution
            for i in range(num_layers - num_dilations, num_layers):
                layer = _DenseLayer(
                        num_input_features + i * growth_rate,
                        growth_rate=growth_rate,
                        bn_size=bn_size,
                        drop_rate=drop_rate,
                        padding=dilations[i - num_layers + num_dilations],
                        dilation=dilations[i - num_layers + num_dilations]
                        )
                self.add_module('atrouslayer%d' % (i + 1), layer)
        else:
            for i in range(num_layers):
                layer = _DenseLayer(
                        num_input_features + i * growth_rate,
                        growth_rate=growth_rate,
                        bn_size=bn_size,
                        drop_rate=drop_rate,
                        padding=1,
                        dilation=1
                        )
                self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.items():
            new_features= layer(features)
            features.append(new_features)

        return torch.cat(features, 1)

class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features, is_pool=True):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm3d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv3d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False))
        if is_pool:
            self.add_module('pool', nn.AvgPool3d(kernel_size=2, stride=2))

class UpSampleBlock(nn.Module):
    def __init__(self, in_planes, out_planes, scale_factor=(2, 2, 2)):
        super(UpSampleBlock, self).__init__()
        self.scale_factor = scale_factor
        self.up = nn.functional.interpolate
        self.conv = nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm3d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, out=None):
        #x = self.up(x, scale_factor=self.scale_factor, mode='trilinear', align_corners=True)
        x = self.up(x, scale_factor=self.scale_factor, mode='nearest')
        if out is not None:
            x = torch.cat([x, out], 1)
        x = self.relu(self.bn(self.conv(x)))

        return x

class AtrousDenseNet(nn.Module):
    """
    DenseNet 3D with atrous convolution

    Args:
    growth_rate (int): how many filters to add each layer ('k' in paper)
    block_config (list of 4 ints): how many layers in each pooling block
    num_init_features (int): the number of filters to learn in the first convolution layer
    bn_size (int): multiplicative factor for number of bottle neck layers. (i.e. bn_size * k features in the bottleneck layer)
    drop_rate (float): dropout rate after each dense layer
    num_classes (int): number of segmentation class
    dilations (list of 4 ints): atrous rate in atrous convolution 
    """

    def __init__(self, in_planes=1, out_planes=2, growth_rate=6, block_config=(6, 12, 12, 6), num_init_features=64, bn_size=4, drop_rate=0., use_dilation=True):
        super(AtrousDenseNet, self).__init__()

        # Input convolution 
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv3d(in_planes, num_init_features, kernel_size=5, 
                                stride=1, padding=2, bias=False)),
            ('norm0', nn.BatchNorm3d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool3d(kernel_size=3, stride=(1, 2, 2), padding=1))
            ]))

        # Dense Blocks
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                    num_layers=num_layers,
                    num_input_features=num_features, 
                    bn_size=bn_size,
                    growth_rate=growth_rate,
                    drop_rate=drop_rate,
                    output_stride=2 ** (i + 1),
                    use_dilation=use_dilation,
                    )
            #print(2 ** (i + 1))
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                if i == len(block_config) - 2:
                    trans = _Transition(num_input_features=num_features,
                                        num_output_features=num_features // 2, is_pool=False)
                    self.features.add_module('transition%d' % (i + 1), trans)
                    num_features = num_features // 2
                else:
                    trans = _Transition(num_input_features=num_features,
                                        num_output_features=num_features // 2)
                    self.features.add_module('transition%d' % (i + 1), trans)
                    num_features = num_features // 2
        self.features.add_module('norm5', nn.BatchNorm3d(num_features))

        # building atrous dense unet
        self.input_block = self.features[:4]
        self.dense_block_1 = self.features[4]
        self.transition_block_1 = self.features[5]
        self.dense_block_2 = self.features[6]
        self.transition_block_2 = self.features[7]
        self.dense_block_3 = self.features[8]
        self.transition_block_3 = self.features[9]
        self.dense_block_4 = self.features[10]
        self.transition_block_4 = self.features[11]

        self.up_1 = UpSampleBlock(32, 16, scale_factor=(1, 2, 2)) 
        self.up_2 = UpSampleBlock(64, 32)
        self.up_3 = UpSampleBlock(102, 64)

        self.output_block = nn.Conv3d(16, out_planes, kernel_size=1, stride=1, padding=0, bias=False)

        # weight init
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        #print('x.shape', x.shape)
        x_64 = self.input_block(x)
        x_64 = self.dense_block_1(x_64)
        #print('x_64.shape', x_64.shape)
        x_32 = self.transition_block_1(x_64)
        x_32 = self.dense_block_2(x_32)
        #print('x_32.shape', x_32.shape)
        x_16 = self.transition_block_2(x_32)
        x_16 = self.dense_block_3(x_16)
        #print('x_16.shape', x_16.shape)
        x_8 = self.transition_block_3(x_16)
        x_8 = self.dense_block_4(x_8)
        out = self.transition_block_4(x_8)
        #out = self.features(x)
        #print('out.shape', out.shape)
        #out = self.up_4(out)
        #print('up_4.shape', out.shape)
        out = self.up_3(out)
        #print('up_3.shape', out.shape)
        out = self.up_2(out)
        #print('up_2.shape', out.shape)
        out = self.up_1(out)
        #print('up_1.shape', out.shape)
        out = self.output_block(out)
        #print('out.shape', out.shape)
        return out
