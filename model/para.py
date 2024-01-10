# 几个参数

class GBuffer:
    def __init__(self, depth, normals,albedo):
        self.depth = depth
        self.normals = normals
        self.albedo = albedo

class Frame:
    def __init__(self, lr_depth_data, lr_normals_data,hr_depth_data, hr_normals_data, irradiance, motion_vector,albedo, shape):
        self.irradiance = irradiance
        self.motion_vector = motion_vector
        self.lr_gbuffer = GBuffer(depth=lr_depth_data, normals=lr_normals_data, albedo)
        self.hr_gbuffer = GBuffer(depth=hr_depth_data, normals=hr_normals_data, albedo)
        self.shape = shape