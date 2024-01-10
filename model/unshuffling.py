
import time
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import para


#  transforming a map of shape [𝐶, 𝐻 ∗ 𝑟,𝑊 ∗ 𝑟] into an LR map of shape [𝐶 ∗ 𝑟 ∗ 𝑟, 𝐻,𝑊 ]
class PixelUnshufflingModule(nn.Module):
    def __init__(self, scale):
        super(PixelUnshufflingModule, self).__init__()
        self.scale = scale

    def forward(self, frame_i):
        '''
        _, _, hh, ww = frame_i.hr_gbuffer.size()  # 输入的 HR 特征图，[C, H, W]
        lr_unshuffled = frame_i.hr_gbuffer.view(-1, 1, hh // self.scale, self.scale, ww // self.scale, self.scale)
        lr_unshuffled = lr_unshuffled.permute(0, 1, 3, 5, 2, 4).contiguous().view(-1, 1, hh // self.scale, ww // self.scale)  # lr_unshuffled，[C * self.scale * self.scale, hh // self.scale, ww // self.scale]
        
        # FIXME: 是否需要拼接？
        # fused_features = torch.cat([lr_unshuffled, lr_inputs], dim=1) # 拼接，得到的 fused_features 的形状为 [C * self.scale * self.scale + C, hh // self.scale, ww // self.scale]

        return lr_unshuffled
        '''
        
        return nn.functional.pixel_unshuffle(frame_i.hr_gbuffer, self.scale) #使用pytorch中现成的pixel unshuffling代码

    
class PixelShufflingModule(nn.Module):
    def __init__(self, scale):
        super(PixelShufflingModule, self).__init__()
        self.scale = scale

    def forward(self, feature_map, skip_connection):
        concatenated_features = torch.cat([feature_map, skip_connection], dim=1)
        return nn.functional.pixel_shuffle(concatenated_features, self.scale) #使用pytorch中现成的pixel shuffling代码

