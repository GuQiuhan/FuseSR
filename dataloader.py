import os
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import imageio
import para
import cv2

class CustomDataset(Dataset):
    def __init__(self, data_root, start_frame, end_frame):
        self.data_root = data_root
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.transform = transforms.ToTensor()
        self.lw,self.lh=480, 270 #low-resolution:270P

    def __len__(self):
        return self.end_frame - self.start_frame + 1

    # 现有的demodulation
    def demodulation(self,albedo, irradiance):
        pos=(albedo==0.)
        albedo[pos]=1.0
        irradiance=irradiance/albedo
        irradiance[pos]=0.0
        return irradiance

    def load_frame_data(self, frame_idx, data_type): # 使用1080P的数据
        if data_type == 'PreTonemapHDRColor':
            subdir = 'GT-1080P'
        else:
            subdir = 'GBuffer-1080P'

        frame_path = os.path.join(self.data_root, f'{subdir}/Bunker{data_type}.{frame_idx:04d}.exr')
        img_data = imageio.imread(frame_path)
        return img_data


    def __getitem__(self, idx):
        frame_idx = self.start_frame + idx

        # Load data for the current frame (frame_i)
        irradiance_i = self.load_frame_data(frame_idx, 'PreTonemapHDRColor')
        motion_vector_i = self.load_frame_data(frame_idx, 'MotionVectorAndMetallicAndRoughness')
        motion_vector_i=motion_vector_i[:,:,0:2]
        albedo_i=self.load_frame_data(frame_idx, 'BaseColor')
        irradiance_i=self.demodulation(albedo_i,irradiance_i)
        hr_depth_i = self.load_frame_data(frame_idx, 'WorldNormalAndSceneDepth')
        lr_depth_i=cv2.resize(hr_depth_i, (self.lw,self.lh), interpolation=cv2.INTER_AREA) # 获得低分辨率
        hr_depth_i=hr_depth_i[:,:,2:4] # 切片
        lr_depth_i=lr_depth_i[:,:,2:4]
        hr_normals_i = self.load_frame_data(frame_idx, 'WorldNormalAndSceneDepth')
        lr_normals_i=cv2.resize(hr_normals_i, (self.lw,self.lh), interpolation=cv2.INTER_AREA)
        hr_normals_i=hr_normals_i[:,:,0:2]
        lr_normals_i=lr_normals_i[:,:,0:2]

        #frame_i = Frame(lr_depth_i, lr_normals_i, hr_depth_i, hr_normals_i, irradiance_i, motion_vector_i,albedo_i,motion_vector_i.shape)
        
        frame_i = {
        'lr_depth': lr_depth_i,
        'lr_normals': lr_normals_i,
        'hr_depth': hr_depth_i,
        'hr_normals': hr_normals_i,
        'irradiance': irradiance_i,
        'motion_vector': motion_vector_i,
        'albedo': albedo_i,
        }

        # Load data for the previous two frames (frame_i_1 and frame_i_2)
        frame_idx_1 = frame_idx - 1

        irradiance_i_1 = self.load_frame_data(frame_idx_1, 'PreTonemapHDRColor')
        motion_vector_i_1 = self.load_frame_data(frame_idx_1, 'MotionVectorAndMetallicAndRoughness')
        motion_vector_i_1=motion_vector_i_1[:,:,0:2]
        albedo_i_1=self.load_frame_data(frame_idx_1, 'BaseColor')
        irradiance_i_1=self.demodulation(albedo_i_1,irradiance_i_1)
        hr_depth_i_1 = self.load_frame_data(frame_idx_1, 'WorldNormalAndSceneDepth')
        lr_depth_i_1=cv2.resize(hr_depth_i_1, (self.lw,self.lh), interpolation=cv2.INTER_AREA) # 获得低分辨率
        hr_depth_i_1=hr_depth_i_1[:,:,2:4] # 切片
        lr_depth_i_1=lr_depth_i_1[:,:,2:4]
        hr_normals_i_1 = self.load_frame_data(frame_idx_1, 'WorldNormalAndSceneDepth')
        lr_normals_i_1=cv2.resize(hr_normals_i_1, (self.lw,self.lh), interpolation=cv2.INTER_AREA)
        hr_normals_i_1=hr_normals_i_1[:,:,0:2]
        lr_normals_i_1=lr_normals_i_1[:,:,0:2]
  
        frame_i_1 = {
        'lr_depth': lr_depth_i_1,
        'lr_normals': lr_normals_i_1,
        'hr_depth': hr_depth_i_1,
        'hr_normals': hr_normals_i_1,
        'irradiance': irradiance_i_1,
        'motion_vector': motion_vector_i_1,
        'albedo': albedo_i_1,
        }

        frame_idx_2 = frame_idx - 2

        irradiance_i_2 = self.load_frame_data(frame_idx_2, 'PreTonemapHDRColor')
        motion_vector_i_2 = self.load_frame_data(frame_idx_2, 'MotionVectorAndMetallicAndRoughness')
        motion_vector_i_2=motion_vector_i_2[:,:,0:2]
        albedo_i_2=self.load_frame_data(frame_idx_2, 'BaseColor')
        irradiance_i_2=self.demodulation(albedo_i_2,irradiance_i_2)
        hr_depth_i_2 = self.load_frame_data(frame_idx_2, 'WorldNormalAndSceneDepth')
        lr_depth_i_2=cv2.resize(hr_depth_i_2, (self.lw,self.lh), interpolation=cv2.INTER_AREA) # 获得低分辨率
        hr_depth_i_2=hr_depth_i_2[:,:,2:4] # 切片
        lr_depth_i_2=lr_depth_i_2[:,:,2:4]
        hr_normals_i_2 = self.load_frame_data(frame_idx_2, 'WorldNormalAndSceneDepth')
        lr_normals_i_2=cv2.resize(hr_normals_i_2, (self.lw,self.lh), interpolation=cv2.INTER_AREA)
        hr_normals_i_2=hr_normals_i_2[:,:,0:2]
        lr_normals_i_2=lr_normals_i_2[:,:,0:2]
  
        frame_i_2 = {
        'lr_depth': lr_depth_i_2,
        'lr_normals': lr_normals_i_2,
        'hr_depth': hr_depth_i_2,
        'hr_normals': hr_normals_i_2,
        'irradiance': irradiance_i_2,
        'motion_vector': motion_vector_i_2,
        'albedo': albedo_i_2,
        }

        sample = {'frame_i': frame_i,'frame_i_1': frame_i_1, 'frame_i_2': frame_i_2}
       
        return sample
'''    
# 使用dataloader:
    
data=CustomDataset('Bunker_1',302,302)

# 创建DataLoader，指定batch_size
#print(data.__getitem__(0).shape)
dataloader = DataLoader(data, batch_size=1, shuffle=True)
for batch_data in dataloader:
    
    print("irradiance:",batch_data['frame_i']['irradiance'].shape) # 返回[batch_size, h, w, channel]
    print("lr_depth:",batch_data['frame_i']['lr_depth'].shape)
    print("lr_normals:",batch_data['frame_i']['lr_normals'].shape)
    print("hr_depth:",batch_data['frame_i']['hr_depth'].shape)
    print("hr_normals:",batch_data['frame_i']['hr_normals'].shape)
    print("motion_vector:",batch_data['frame_i']['motion_vector'].shape)
    print("albedo:",batch_data['frame_i']['albedo'].shape)

# 各个shape 
irradiance: torch.Size([1, 1080, 1920, 4])
lr_depth: torch.Size([1, 270, 480, 2])
lr_normals: torch.Size([1, 270, 480, 2])
hr_depth: torch.Size([1, 1080, 1920, 2])
hr_normals: torch.Size([1, 1080, 1920, 2])
motion_vector: torch.Size([1, 1080, 1920, 2])
albedo: torch.Size([1, 1080, 1920, 4])
    
'''