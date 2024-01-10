import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self):
        super(ResidualBlock, self).__init__()

        #FIXME:论文中没有明说这些参数的值(32-48-64-96)
        self.in_channels=48
        self.out_channels=48
        self.kernel_size=3
        self.stride=1
        self.padding=1

        self.conv1 = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, self.kernel_size, self.stride, self.padding)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += residual # 做pixel-wise add
        return out

# 输入拼接后的feature map，输出 an LR fused feature map
class FusionNetwork(nn.Module):
    def __init__(self):
        super(FusionNetwork, self).__init__()

        # 六个残差模块
        self.fusion_module = nn.Sequential(
            ResidualBlock(),
            ResidualBlock(),
            ResidualBlock(),
            ResidualBlock(),
            ResidualBlock(),
            ResidualBlock()
        )

    def forward(self, feature_map):
        fused_feature_map = self.fusion_module(feature_map)

        return fused_feature_map
