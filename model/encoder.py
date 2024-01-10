
import time
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import para


class EncoderNetwork(nn.Module):
    def __init__(self, in_channels=6, hidden_channels=[64, 64, 32, 24, 24, 32, 64]):
        super(Encoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels[0], kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels[0], hidden_channels[1], kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels[1], hidden_channels[2], kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels[2], hidden_channels[3], kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels[3], hidden_channels[4], kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels[4], hidden_channels[5], kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels[5], hidden_channels[6], kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )


    def forward(self, lr_irradiance, lr_gbuffers, prev_encoded=None, prev_hidden=None):
        x = torch.cat((lr_irradiance, lr_gbuffers), dim=1)

        encoded_features = self.encoder(x)
        skip_connection = encoded_features[0]

        return skip_connection, encoded_features[-1]  # 第一个卷积的输出作为skip_connection,最后一层输出作为fusion input
    

class WarpingModule(nn.Module):
    def __init__(self):
        super(WarpingModule, self).__init__()

    def forward(self, historical_frames, motion_vectors):
        # 假定历史帧格式： [batch_size, num_frames, channels, height, width]
        # 假定运动向量格式： [batch_size, num_frames, 2] representing (dx, dy)
        batch_size, num_frames, channels, height, width = historical_frames.size()

        historical_frames_reshaped = historical_frames.view(batch_size * num_frames, channels, height, width) # 转换历史帧格式为[batch_size * num_frames, channels, height, width]
        warped_frames = self.warp(historical_frames_reshaped, motion_vectors)
        warped_frames = warped_frames.view(batch_size, num_frames, channels, height, width) # 转换回[batch_size, num_frames, channels, height, width]

        return warped_frames
    '''
    # FIXME: 使用常见的双线性插值，论文中缺少对warping的具体描述
    def warp(self, input_frames, motion_vectors):
        # Extract motion vectors along x and y directions
        dx = motion_vectors[:, :, 0].unsqueeze(-1).unsqueeze(-1)
        dy = motion_vectors[:, :, 1].unsqueeze(-1).unsqueeze(-1)

        # Generate grid for interpolation
        grid_x, grid_y = torch.meshgrid(torch.linspace(-1, 1, input_frames.size(-1)),
                                        torch.linspace(-1, 1, input_frames.size(-2)))

        grid_x = grid_x.to(input_frames.device)
        grid_y = grid_y.to(input_frames.device)

        # Apply motion vectors to the grid
        grid_x = grid_x - 2 * dx
        grid_y = grid_y - 2 * dy

        # Normalize grid to [-1, 1]
        grid_x_normalized = (grid_x + 1) / 2 * 2 - 1
        grid_y_normalized = (grid_y + 1) / 2 * 2 - 1

        # Stack grid_x and grid_y to form the final grid
        grid = torch.stack((grid_x_normalized, grid_y_normalized), dim=-1)

        # Perform bilinear interpolation
        warped_frames = nn.functional.grid_sample(input_frames, grid, align_corners=False)

        return warped_frames
    '''
    def warp(self,irradiance, motion_vectors):
        irradiance=irradiance.permute(0, 3, 1, 2)
        motion_vectors=motion_vectors.permute(0, 3, 1, 2)
        n,c,h,w=irradiance.shape
        dx,dy=torch.linspace(-1,1,w),torch.linspace(-1,1,h)
        grid_y,grid_x=torch.meshgrid(dy,dx,indexing='ij')

        grid_x = grid_x.repeat(n, 1, 1) - (2 * motion_vectors[:,1]/(w))
        grid_y = grid_y.repeat(n, 1, 1) + (2 * motion_vectors[:,0]/ (h))

        coord=torch.stack([grid_x,grid_y],dim=-1)
        res=F.grid_sample(irradiance,coord,padding_mode='zeros',align_corners=True)
        return res
    

# TODO:
class AttentionModule(nn.Module):
    def __init__(self, input_channels):
        super(AttentionModule, self).__init__()    
        self.conv_hist1 = nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1) # 卷积和ReLU处理历史帧i-1
        self.conv_hist2 = nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1) # 卷积和ReLU处理历史帧i-2
        self.conv_tanh = nn.Conv2d(input_channels, input_channels, kernel_size=1) # 卷积层，用于tanh
        self.conv_sigmoid = nn.Conv2d(input_channels, 1, kernel_size=1) # 卷积层，用于sigmoid

    def forward(self, frame_i_1, frame_i_2):
        # 𝐴𝑘 = sigmoid(Convv(tanh(Conva (𝐻) + Convb(𝐻))))
        hist_frame1 = F.relu(self.conv_hist1(frame_i_1))
        hist_frame2 = F.relu(self.conv_hist2(frame_i_2))

        hist_frame_combined = hist_frame1 + hist_frame2
        hist_frame_tanh = torch.tanh(self.conv_tanh(hist_frame_combined))
        attention_map = torch.sigmoid(self.conv_sigmoid(hist_frame_tanh))

        return attention_map



# Encoder由三个部分组成，encoder, warping, attention
# 输入三个帧，输出一个feature map和skip connection
class EncoderModule(nn.Module):
    def __init__(self):
        super(EncoderModule, self).__init__()
        self.encoder = EncoderNetwork()
        self.warping = WarpingModule()
        self.attention = AttentionModule()

    def forward(self, frame_i, frame_i_1, frame_i_2): # 当前帧和历史两帧的deformulated irradiance& HR gbuffer
        # Encoder
        skip_connection,lr_feature_maps_i= self.encoder(frame_i.irradiance, frame_i.hr_gbuffer) # Encoder
       
        # Recurrent Unit
        # FIXME:如何保证share network weights for the encoder network for current and historical frames, referred to as “Recurrent Unit”
        _,lr_feature_maps_i_1= self.encoder(frame_i_1.irradiance, frame_i_1.hr_gbuffer) # 历史帧i-1
        _,lr_feature_maps_i_2= self.encoder(frame_i_2.irradiance, frame_i_2.hr_gbuffer) # 历史帧i-2

        # Warping
        warped_features_i_1 = self.warping(lr_feature_maps_i_1, frame_i_1.motion_vector)  
        warped_features_i_2 = self.warping(lr_feature_maps_i_2, frame_i_2.motion_vector)  

        # Attention
        # FIXME: 这里的d是什么参数？--> depth
        attention_map_i_1 = self.attention(frame_i.hr_gbuffer.depth, frame_i_1.hr_gbuffer.depth, frame_i_1.motion_vector) 
        attention_map_i_2 = self.attention(frame_i.hr_gbuffer.depth, frame_i_2.hr_gbuffer.depth, frame_i_2.motion_vector) 

        final_features = attention_map_i_1 * warped_features_i_1 + attention_map_i_2 * warped_features_i_2
        final_features += lr_feature_maps_i

        return final_features, skip_connection # skip connection也一起返回，之后会用到
    
