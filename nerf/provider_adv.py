import os
import cv2
import glob
import json
import tqdm
import numpy as np
from scipy.spatial.transform import Slerp, Rotation
import random
import trimesh

import torch
from torch.utils.data import DataLoader

from .utils import get_rays, create_dodecahedron_cameras

def nerf_matrix_to_ngp(pose, scale=0.33, offset=[0, 0, 0]):
    pose[:3, 3] = pose[:3, 3] * scale + np.array(offset)
    pose = pose.astype(np.float32)
    return pose

def pose_spherical(all_args):

    gamma, theta, phi, radius, x, y = all_args[0], all_args[1], all_args[2], all_args[3], all_args[4], all_args[5]

    trans_t = lambda t, x, y: torch.Tensor([
    [1, 0, 0, x],
    [0, 1, 0, y],
    [0, 0, 1, t],
    [0, 0, 0, 1]]).float()

    rot_phi = lambda phi: torch.Tensor([
    [1, 0, 0, 0],
    [0, np.cos(phi), -np.sin(phi), 0],
    [0, np.sin(phi), np.cos(phi), 0],
    [0, 0, 0, 1]]).float()

    rot_gamma = lambda gamma: torch.Tensor([
    [np.cos(gamma), 0, -np.sin(gamma), 0],
    [0, 1, 0, 0],
    [np.sin(gamma), 0, np.cos(gamma), 0],
    [0, 0, 0, 1]]).float()

    rot_theta = lambda th: torch.Tensor([
    [np.cos(th), -np.sin(th), 0, 0],
    [np.sin(th), np.cos(th), 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]]).float()

    c2w = trans_t(radius, x, y)
    c2w = rot_phi(phi / 180. * np.pi) @ c2w
    c2w = rot_gamma(gamma / 180. * np.pi) @ c2w
    c2w = rot_theta(theta / 180. * np.pi) @ c2w

    return c2w

def rand_poses(size, device, radius=1, theta_range=[60, 120], phi_range=[-60, 60], gamma_range=[-60, 60]):
    # generate random gamma, theta, phi, radius, x, and y
    gamma = np.random.uniform(gamma_range[0], gamma_range[1], size)  # gamma in degrees
    theta = np.random.uniform(theta_range[0], theta_range[1], size)  # theta in degrees
    phi = np.random.uniform(phi_range[0], phi_range[1], size)  # phi in degrees
    radius = np.full(size, radius)  # radius is fixed
    x = np.random.uniform(-0.1, 0.1, size)  # x in range [-1, 1]
    y = np.random.uniform(-0.1, 0.1, size)  # y in range [-1, 1]

    poses = []
    for i in range(size):
        all_args = [gamma[i], theta[i], phi[i], radius[i], x[i], y[i]]
        pose = pose_spherical(all_args)
        poses.append(pose)

    # convert list of poses to tensor
    poses = torch.stack(poses).to(device)
    
    return poses

class NeRFDataset:
    def __init__(self, opt, device, type='train', n_test=10):
        super().__init__()
        
        self.opt = opt
        self.device = device
        self.type = type # train, val, test
        self.downscale = opt.downscale
        self.root_path = opt.path
        self.preload = opt.preload # preload data into GPU
        self.scale = opt.scale # camera radius scale to make sure camera are inside the bounding box.
        self.offset = opt.offset # camera offset
        self.bound = opt.bound # bounding box half length, also used as the radius to random sample poses.
        self.fp16 = opt.fp16 # if preload, load into fp16.

        if self.scale == -1:
            print(f'[WARN] --data_format nerf cannot auto-choose --scale, use 1 as default.')
            self.scale = 1
            
        self.training = self.type in ['train', 'all', 'trainval']

        # auto-detect transforms.json and split mode.
        if os.path.exists(os.path.join(self.root_path, 'transforms.json')):
            self.mode = 'colmap' # manually split, use view-interpolation for test.
        elif os.path.exists(os.path.join(self.root_path, 'transforms_train.json')):
            self.mode = 'blender' # provided split
        else:
            raise NotImplementedError(f'[NeRFDataset] Cannot find transforms*.json under {self.root_path}')

        # load nerf-compatible format data.
        if self.mode == 'colmap':
            with open(os.path.join(self.root_path, 'transforms.json'), 'r') as f:
                transform = json.load(f)
        elif self.mode == 'blender':
            # load all splits (train/valid/test), this is what instant-ngp in fact does...
            if type == 'all':
                transform_paths = glob.glob(os.path.join(self.root_path, '*.json'))
                transform = None
                for transform_path in transform_paths:
                    with open(transform_path, 'r') as f:
                        tmp_transform = json.load(f)
                        if transform is None:
                            transform = tmp_transform
                        else:
                            transform['frames'].extend(tmp_transform['frames'])
            # load train and val split
            elif type == 'trainval':
                with open(os.path.join(self.root_path, f'transforms_train.json'), 'r') as f:
                    transform = json.load(f)
                with open(os.path.join(self.root_path, f'transforms_val.json'), 'r') as f:
                    transform_val = json.load(f)
                transform['frames'].extend(transform_val['frames'])
            # only load one specified split
            else:
                with open(os.path.join(self.root_path, f'transforms_{type}.json'), 'r') as f:
                    transform = json.load(f)

        else:
            raise NotImplementedError(f'unknown dataset mode: {self.mode}')

        # load image size
        if 'h' in transform and 'w' in transform:
            self.H = int(transform['h']) // self.downscale
            self.W = int(transform['w']) // self.downscale
        else:
            # we have to actually read an image to get H and W later.
            self.H = self.W = None
        
        # read images
        frames = np.array(transform["frames"])
        
        # tmp: if time in frames (dynamic scene), only load time == 0
        if 'time' in frames[0]:
            frames = np.array([f for f in frames if f['time'] == 0])
            print(f'[INFO] selecting time == 0 frames: {len(transform["frames"])} --> {len(frames)}')


        # for colmap, manually interpolate a test set.
        if self.mode == 'colmap' and type == 'test':
            
            # choose two random poses, and interpolate between.
            f0, f1 = np.random.choice(frames, 2, replace=False)
            pose0 = nerf_matrix_to_ngp(np.array(f0['transform_matrix'], dtype=np.float32), scale=self.scale, offset=self.offset) # [4, 4]
            pose1 = nerf_matrix_to_ngp(np.array(f1['transform_matrix'], dtype=np.float32), scale=self.scale, offset=self.offset) # [4, 4]
            rots = Rotation.from_matrix(np.stack([pose0[:3, :3], pose1[:3, :3]]))
            slerp = Slerp([0, 1], rots)

            self.poses = []
            self.images = None
            for i in range(n_test + 1):
                ratio = np.sin(((i / n_test) - 0.5) * np.pi) * 0.5 + 0.5
                pose = np.eye(4, dtype=np.float32)
                pose[:3, :3] = slerp(ratio).as_matrix()
                pose[:3, 3] = (1 - ratio) * pose0[:3, 3] + ratio * pose1[:3, 3]
                self.poses.append(pose)

        else:
            # for colmap, manually split a valid set (the first frame).
            if self.mode == 'colmap':
                if type == 'train':
                    frames = frames[1:]
                elif type == 'val':
                    frames = frames[:1]
                # else 'all' or 'trainval' : use all frames
            
            self.poses = []
            self.images = []
            for f in tqdm.tqdm(frames, desc=f'Loading {type} data'):
                f_path = os.path.join(self.root_path, f['file_path'])
                if self.mode == 'blender' and '.' not in os.path.basename(f_path):
                    f_path += '.png' # so silly...

                # there are non-exist paths in fox...
                if not os.path.exists(f_path):
                    print(f'[WARN] {f_path} not exists!')
                    continue
                
                pose = np.array(f['transform_matrix'], dtype=np.float32) # [4, 4]
                pose[0][0] = pose[0][0] + 0.01
                pose = nerf_matrix_to_ngp(pose, scale=self.scale, offset=self.offset)

                image = cv2.imread(f_path, cv2.IMREAD_UNCHANGED) # [H, W, 3] o [H, W, 4]
                if self.H is None or self.W is None:
                    self.H = image.shape[0] // self.downscale
                    self.W = image.shape[1] // self.downscale

                # add support for the alpha channel as a mask.
                if image.shape[-1] == 3: 
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)

                if image.shape[0] != self.H or image.shape[1] != self.W:
                    image = cv2.resize(image, (self.W, self.H), interpolation=cv2.INTER_AREA)

                self.poses.append(pose)
                self.images.append(image)
            
        self.poses = torch.from_numpy(np.stack(self.poses, axis=0)) # [N, 4, 4]
        if self.images is not None:
            self.images = torch.from_numpy(np.stack(self.images, axis=0).astype(np.uint8)) # [N, H, W, C]
        
        # calculate mean radius of all camera poses
        self.radius = self.poses[:, :3, 3].norm(dim=-1).mean(0).item()
        #print(f'[INFO] dataset camera poses: radius = {self.radius:.4f}, bound = {self.bound}')
        self.poses_rnd = rand_poses(len(self.poses), self.device, self.radius, theta_range=[-60, 120], phi_range=[-60, 120], gamma_range=[-60, 120])
            
        # load intrinsics
        if 'fl_x' in transform or 'fl_y' in transform:
            fl_x = (transform['fl_x'] if 'fl_x' in transform else transform['fl_y']) / self.downscale
            fl_y = (transform['fl_y'] if 'fl_y' in transform else transform['fl_x']) / self.downscale
        elif 'camera_angle_x' in transform or 'camera_angle_y' in transform:
            # blender, assert in radians. already downscaled since we use H/W
            fl_x = self.W / (2 * np.tan(transform['camera_angle_x'] / 2)) if 'camera_angle_x' in transform else None
            fl_y = self.H / (2 * np.tan(transform['camera_angle_y'] / 2)) if 'camera_angle_y' in transform else None
            if fl_x is None: fl_x = fl_y
            if fl_y is None: fl_y = fl_x
        else:
            raise RuntimeError('Failed to load focal length, please check the transforms.json!')

        cx = (transform['cx'] / self.downscale) if 'cx' in transform else (self.W / 2.0)
        cy = (transform['cy'] / self.downscale) if 'cy' in transform else (self.H / 2.0)
    
        self.intrinsics = np.array([fl_x, fl_y, cx, cy])

        # perspective projection matrix
        self.near = self.opt.min_near
        self.far = 1000 # infinite

        y = self.H / (2.0 * fl_y)

        aspect =  self.W / self.H
        self.projection = np.array([[1/(y*aspect), 0, 0, 0], 
                                    [0, -1/y, 0, 0],
                                    [0, 0, -(self.far+self.near)/(self.far-self.near), -(2*self.far*self.near)/(self.far-self.near)],
                                    [0, 0, -1, 0]], dtype=np.float32)

        self.projection = torch.from_numpy(self.projection)

        self.projection_rnd = np.array([[1/(y*aspect), 0, 0, 0], 
                            [0, -1/y, 0, 0],
                            [0, 0, -(self.far+self.near)/(self.far-self.near), -(2*self.far*self.near)/(self.far-self.near)],
                            [0, 0, -1, 0]], dtype=np.float32)

        self.projection_rnd = torch.from_numpy(self.projection_rnd)
        self.mvps_rnd = self.projection_rnd.unsqueeze(0) @ torch.inverse(self.poses_rnd.to('cpu'))
        self.mvps = self.projection.unsqueeze(0) @ torch.inverse(self.poses)
    
        # tmp: dodecahedron_cameras for mesh visibility test
        dodecahedron_poses = create_dodecahedron_cameras()
        self.dodecahedron_poses = torch.from_numpy(dodecahedron_poses.astype(np.float32)) # [N, 4, 4]
        self.dodecahedron_mvps = self.projection.unsqueeze(0) @ torch.inverse(self.dodecahedron_poses)

        if self.preload:
            self.poses = self.poses.to(self.device)
            if self.images is not None:
                self.images = self.images.to(self.device)
            self.projection = self.projection.to(self.device)
            self.mvps = self.mvps.to(self.device)
            self.mvps_rnd = self.mvps_rnd.to(self.device)



    def collate(self, index):

        B = len(index) # a list of length 1

        results = {'H': self.H, 'W': self.W}

        if self.training and self.opt.stage == 0:
            # randomly sample over images too
            num_rays = self.opt.num_rays

            if self.opt.random_image_batch:
                index = torch.randint(0, len(self.poses), size=(num_rays,), device=self.device)

        else:
            num_rays = -1
        if random.random() < 0.5:
            results['ifrandom'] = 1
            poses = self.poses_rnd[index].to(self.device)
            # print(poses[0])
        else:
            # print('not random')
            results['ifrandom'] = 0
            poses = self.poses[index].to(self.device) # [N, 4, 4]
            # print(poses[0])
        rays = get_rays(poses, self.intrinsics, self.H, self.W, num_rays, self.opt.patch_size)

        results['rays_o'] = rays['rays_o']
        results['rays_d'] = rays['rays_d']
        results['index'] = index

        if self.opt.stage > 0:
            if results['ifrandom']:
                mvp = self.mvps_rnd[index].to(self.device)
            else:
                mvp = self.mvps[index].to(self.device)
            results['mvp'] = mvp

        if self.images is not None:
            
            if self.training and self.opt.stage == 0:
                images = self.images[index, rays['j'], rays['i']].float().to(self.device) / 255 # [N, 3/4]
            else:
                images = self.images[index].squeeze(0).float().to(self.device) / 255 # [H, W, 3/4]

            if self.training:
                C = self.images.shape[-1]
                images = images.view(-1, C)

            results['images'] = images
            
        return results

    def dataloader(self):
        size = len(self.poses)
        loader = DataLoader(list(range(size)), batch_size=1, collate_fn=self.collate, shuffle=self.training, num_workers=0)
        loader._data = self # an ugly fix... we need to access error_map & poses in trainer.
        loader.has_gt = self.images is not None
        return loader