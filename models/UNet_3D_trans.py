import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class InConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(InConv, self).__init__()
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool3d(2, 2),# 池化核和步长大小
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x

class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, 1)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        # x = self.sigmoid(x)
        return x

# 上采样块
class Up(nn.Module):
    def __init__(self, in_ch, skip_ch,out_ch):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose3d(in_ch, in_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch+skip_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class UNet(nn.Module):
    def __init__(self, in_channels, num_classes,fea):
        super(UNet, self).__init__()
        features = fea

        self.inc = InConv(in_channels, features[0]) # 先增大特征空间 [1,1,16,256,256]--->[1,32,16,256,256]

        # 四个下采样层 先进行最大池化 再使用两个卷积
        self.down1 = Down(features[0], features[1]) # [1,32,16,256,256]--->[1,64,]
        self.down2 = Down(features[1], features[2])
        self.down3 = Down(features[2], features[3])
        self.down4 = Down(features[3], features[3])

        self.up1 = Up(features[3], features[3], features[2])
        self.up2 = Up(features[2], features[2], features[1])
        self.up3 = Up(features[1], features[1], features[0])
        self.up4 = Up(features[0], features[0], features[0])
        self.outc = OutConv(features[0], num_classes)

    def forward(self, x): # [1,1,16,256,256]
        x1 = self.inc(x) # [1, 32, 16, 256, 256]

        x2 = self.down1(x1) # [1, 64, 8, 128, 128]
        x3 = self.down2(x2) # [1, 128, 4, 64, 64]
        x4 = self.down3(x3) # [1, 256, 2, 32, 32]
        x5 = self.down4(x4) # [1, 256, 1, 16, 16]

        # 针对x5输出添加注意力块


        x = self.up1(x5, x4) # [1, 128, 2, 32, 32]
        x = self.up2(x, x3) # [1, 64, 4, 64, 64]
        x = self.up3(x, x2) # [1, 32, 8, 128, 128]
        x = self.up4(x, x1) # [1, 32, 16, 256, 256]
        x = self.outc(x) # [1, 1, 16, 256, 256]

        return x


if __name__ == '__main__':

    x = torch.randn(1, 1, 16, 256, 256)

    features = [32, 64, 128, 256]

    net = UNet(in_channels=1, num_classes=1, fea=features)

    from thop import profile

    flops, params = profile(net, inputs=(x,))

    print(flops / 1e9, params / 1e6)
