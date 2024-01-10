import torch
import torch.nn as nn
import encoder
import fusion
import para
import unshuffling


def make_model(args, parent=False):
    return RCAN(args)

# 最后的3x3 conv layer
class ConvModule(nn.Module):
    def __init__(self):
        super(UpscalingModule, self).__init__()
        #FIXME: 参数未确定
        self.in_channels=3
        self.out_channels=3

        self.conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, upscaled_feature_map):
        rgb_prediction = self.conv(upscaled_feature_map)

        return rgb_prediction
    
class HNet(nn.Module):
    def __init__(self,scale):
        super(HNet, self).__init__()
        self.scale=scale
        self.pixel_unshuffle = unshuffling.PixelUnshufflingModule(scale)
        self.encoder = encoder.EncoderModule()
        self.fusion_network = fusion.FusionNetwork()  
        self.pixel_shuffle = unshuffling.PixelShufflingModule(scale)
        self.conv33=ConvModule()

    def forward(self, frame_i, frame_i_1,frame_i_2):

        unshuffled_hr_map = self.pixel_unshuffle(frame_i.hr_gbuffer)
        final_features, skip_connection=self.encoder(frame_i, frame_i_1,frame_i_2)
        concatenated_features = torch.cat([final_features, unshuffled_hr_map], dim=1) 
        fusion_feature_map = self.fusion_network(concatenated_features)

        upscale_feature_map=self.pixel_shuffle(fusion_feature_map,skip_connection)
 
        rgb_prediction=self.conv33(upscale_feature_map)

        return rgb_prediction*frame_i.hr_gbuffer # 返回像素乘

