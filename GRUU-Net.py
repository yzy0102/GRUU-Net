import torch
import torch.nn as nn
import torch.nn.functional as F
class _DenseLayer(nn.Sequential):
    """Basic unit of DenseBlock (using bottleneck layer) """
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module("norm1", nn.BatchNorm2d(num_input_features))
        self.add_module("relu1", nn.ReLU(inplace=True))
        self.add_module("conv1", nn.Conv2d(num_input_features, bn_size*growth_rate,
                                           kernel_size=1, stride=1, bias=False))
        self.add_module("norm2", nn.BatchNorm2d(bn_size*growth_rate))
        self.add_module("relu2", nn.ReLU(inplace=True))
        self.add_module("conv2", nn.Conv2d(bn_size*growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1, bias=False))
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)
    
class _DenseBlock(nn.Sequential):
    """DenseBlock"""
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features+i*growth_rate, growth_rate, bn_size,
                                drop_rate)
            self.add_module("denselayer%d" % (i+1,), layer)
            
class _Transition(nn.Sequential):
    """Transition layer between two adjacent DenseBlock"""
    def __init__(self, num_input_feature, num_output_features):
        super(_Transition, self).__init__()
        self.add_module("norm", nn.BatchNorm2d(num_input_feature))
        self.add_module("relu", nn.ReLU(inplace=True))
        self.add_module("conv", nn.Conv2d(num_input_feature, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        #self.add_module("pool", nn.AvgPool2d(2, stride=2))
        
class DenseNet(nn.Module):
    "DenseNet-BC model"
    def __init__(self, growth_rate=32, k = 3, in_features=32, out_features=64, bn_size=1, drop_rate=0):

        super(DenseNet, self).__init__()
        
        self.features = nn.Sequential()

        num_features = in_features
        i = 0
        num_layers = k
        
        block = _DenseBlock(num_layers, num_features, bn_size, growth_rate, drop_rate)
        self.features.add_module("denseblock%d" % (i + 1), block)
        num_features += num_layers*growth_rate

        transition = _Transition(num_features, out_features)
        self.features.add_module("transition%d" % (i + 1), transition)
        num_features = int(out_features)
        # final bn+ReLU
        self.features.add_module("norm5", nn.BatchNorm2d(num_features))
        self.features.add_module("relu5", nn.ReLU(inplace=True))

        # params initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        return features
    
class ConvGRU(nn.Module):
    def __init__(self, x_channels=64, channels=32):
        super(ConvGRU, self).__init__()
        self.channels = channels
        self.x_channels = x_channels

        self.conv_x_z = nn.Conv2d(in_channels=self.x_channels, out_channels=self.channels, kernel_size=3, stride=1, padding=1)
        self.conv_h_z = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=3, stride=1, padding=1)
        self.conv_x_r = nn.Conv2d(in_channels=self.x_channels, out_channels=self.channels, kernel_size=3, stride=1, padding=1)
        
        self.conv_h_r = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=3, stride=1, padding=1)
        self.conv =  nn.Conv2d(in_channels=self.x_channels, out_channels=self.channels, kernel_size=3, stride=1, padding=1)
        
        self.conv_u =  nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=3, stride=1, padding=1)
        #self.conv_out = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=3, stride=1, padding=1)
        self.lReLU = nn.LeakyReLU(0.2)

    def forward(self, x, h_t_1):
        """GRU卷积流程
        args:
            x: input
            h_t_1: 上一层的隐含层输出值
        shape：
            x: [in_channels, channels, width, lenth]
        """  
        z_t = F.sigmoid(self.conv_x_z(x) + self.conv_h_z(h_t_1))
        r_t = F.sigmoid((self.conv_x_r(x) + self.conv_h_r(h_t_1)))
        h_hat_t = self.lReLU(self.conv(x) + self.conv_u(torch.mul(r_t, h_t_1)))
        
        
        h_t = torch.mul((1 - z_t), h_t_1) + torch.mul(z_t, h_hat_t)
        #y = self.conv_out(h_t)
        return h_t

class FRDU(nn.Module):
    def __init__(self, in_channels, channels, factor=2):
        super(FRDU, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.factor = factor
        self.convNorm1 = nn.Sequential(
            nn.Conv2d(in_channels+32, channels, 1),
            nn.BatchNorm2d(channels)
        )
        
        self.convNorm2 = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels)
        )    
        
        self.denseLayer = DenseNet(k = 3, in_features=channels, out_features=channels, bn_size=2)
        self.ConvGRU = ConvGRU(x_channels=channels)
        
        
    def forward(self, o_t_1, h_t_1):
        """
        o_t_t: Ot-1输入
        h_t_1: GRU的输出h_t_1
        """
        h_t_ori = h_t_1
        # 原文: We found that using bilinear interpolation instead of max pooling 
        # decreased the stability of the training.
        h_t_1 = F.interpolate(h_t_1 , scale_factor=1/self.factor ,mode='bilinear')
        
        o_t_1 = self.convNorm1(torch.cat([o_t_1, h_t_1], 1))
        
        o_t = self.denseLayer(o_t_1)
        x_t = self.convNorm2(o_t)
        x_t = F.interpolate(x_t , scale_factor=self.factor ,mode='bilinear')
        h_t = self.ConvGRU(x_t, h_t_ori)
        
        return o_t, h_t

class GRUU_Net(nn.Module):
    def __init__(self, num_classes=2):
        super(GRUU_Net, self).__init__()
        self.input = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2)
        )
        self.FRDU_1 = FRDU(32, 64,factor=2)
        self.FRDU_2 = FRDU(64, 128,factor=4)
        self.FRDU_3 = FRDU(128, 256,factor=8)
        self.FRDU_4 = FRDU(256, 512,factor=16)
        self.FRDU_5 = FRDU(512, 512,factor=32)
        self.FRDU_6 = FRDU(512, 256,factor=16)
        self.FRDU_7 = FRDU(256, 128,factor=8)
        self.FRDU_8 = FRDU(128, 64,factor=4)
        self.FRDU_9 = FRDU(64, 32,factor=2)
    
        self.Resblock = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2)
        )
        
        self.cls_seg = nn.Conv2d(32, num_classes, 3, padding=1)
        
    def forward(self, x):
        x = self.input(x)

        o_t, h_t = self.FRDU_1(o_t_1 = nn.MaxPool2d(2)(x), h_t_1 = x)
        o_t, h_t = self.FRDU_2(o_t_1 = nn.MaxPool2d(2)(o_t), h_t_1 = h_t)
        o_t, h_t = self.FRDU_3(o_t_1 = nn.MaxPool2d(2)(o_t), h_t_1 = h_t)
        o_t, h_t = self.FRDU_4(o_t_1 = nn.MaxPool2d(2)(o_t), h_t_1 = h_t)
        o_t, h_t = self.FRDU_5(o_t_1 = nn.MaxPool2d(2)(o_t), h_t_1 = h_t)

        o_t, h_t = self.FRDU_6(o_t_1 = F.interpolate(o_t, scale_factor=2, mode="bilinear"), h_t_1 = h_t)
        o_t, h_t = self.FRDU_7(o_t_1 = F.interpolate(o_t, scale_factor=2, mode="bilinear"), h_t_1 = h_t)
        o_t, h_t = self.FRDU_8(o_t_1 = F.interpolate(o_t, scale_factor=2, mode="bilinear"), h_t_1 = h_t)
        o_t, h_t = self.FRDU_9(o_t_1 = F.interpolate(o_t, scale_factor=2, mode="bilinear"), h_t_1 = h_t)
        
        h_t = self.Resblock(h_t) + h_t
        out = self.cls_seg(h_t)
        return out


if __name__ == "__main__":
        Net = GRUU_Net(3)
        o_t_1 = torch.randn((4,3,224,224))
        out = Net(o_t_1)