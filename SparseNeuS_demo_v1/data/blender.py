import torch
from torch.utils.data import Dataset
import json
import numpy as np
import os
from PIL import Image
from torchvision import transforms as T
from kornia import create_meshgrid
from models.rays import gen_rays_from_single_image, gen_random_rays_from_single_image
import cv2 as cv
from data.scene import get_boundingbox


def get_ray_directions(H, W, focal, center=None):
    """
    Get ray directions for all pixels in camera coordinate.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems
    Inputs:
        H, W, focal: image height, width and focal length
    Outputs:
        directions: (H, W, 3), the direction of the rays in camera coordinate
    """
    grid = create_meshgrid(H, W, normalized_coordinates=False)[0]
    i, j = grid.unbind(-1)
    # the direction here is without +0.5 pixel centering as calibration is not so accurate
    # see https://github.com/bmild/nerf/issues/24
    cent = center if center is not None else [W / 2, H / 2]
    directions = torch.stack([(i - cent[0]) / focal[0], (j - cent[1]) / focal[1], torch.ones_like(i)], -1)  # (H, W, 3)

    return directions

def get_rays(directions, c2w):
    """
    Get ray origin and normalized directions in world coordinate for all pixels in one image.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems
    Inputs:
        directions: (H, W, 3) precomputed ray directions in camera coordinate
        c2w: (3, 4) transformation matrix from camera coordinate to world coordinate
    Outputs:
        rays_o: (H*W, 3), the origin of the rays in world coordinate
        rays_d: (H*W, 3), the normalized direction of the rays in world coordinate
    """
    # Rotate ray directions from camera coordinate to the world coordinate
    rays_d = directions @ c2w[:3, :3].T  # (H, W, 3)
    # rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    # The origin of all rays is the camera origin in world coordinate
    rays_o = c2w[:3, 3].expand(rays_d.shape)  # (H, W, 3)

    rays_d = rays_d.view(-1, 3)
    rays_o = rays_o.view(-1, 3)

    return rays_o, rays_d


def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()  # ? why need transpose here
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose  # ! return cam2world matrix here


class BlenderDataset(Dataset):
    def __init__(self, root_dir, split, scan_id, n_views, train_img_idx=[], test_img_idx=[],
                 img_wh=[800, 800], clip_wh=[0, 0], original_img_wh=[800, 800],
                 N_rays=512, h_patch_size=5, near=2.0, far=6.0):
        self.root_dir = root_dir
        self.split = split
        self.img_wh = img_wh
        self.clip_wh = clip_wh
        self.define_transforms()
        self.train_img_idx = train_img_idx
        self.test_img_idx = test_img_idx
        self.N_rays = N_rays
        self.h_patch_size = h_patch_size  # used to extract patch for supervision
        self.n_views = n_views
        self.near, self.far = near, far
        self.blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        with open(os.path.join(self.root_dir, f"transforms_{self.split}.json"), 'r') as f:
            self.meta = json.load(f)


        self.read_meta(near, far)
        # import ipdb; ipdb.set_trace()
        self.raw_near_fars = np.stack([np.array([self.near, self.far]) for i in range(len(self.meta['frames']))])


        # ! estimate scale_mat
        self.scale_mat, self.scale_factor = self.cal_scale_mat(
            img_hw=[self.img_wh[1], self.img_wh[0]],
            intrinsics=self.all_intrinsics[self.train_img_idx],
            extrinsics=self.all_w2cs[self.train_img_idx],
            near_fars=self.raw_near_fars[self.train_img_idx],
            factor=1.1)
        # self.scale_mat = np.eye(4)
        # self.scale_factor = 1.0
        # import ipdb; ipdb.set_trace()
        # * after scaling and translation, unit bounding box
        self.scaled_intrinsics, self.scaled_w2cs, self.scaled_c2ws, \
        self.scaled_affine_mats, self.scaled_near_fars = self.scale_cam_info()

        self.bbox_min = np.array([-1.0, -1.0, -1.0])
        self.bbox_max = np.array([1.0, 1.0, 1.0])
        self.partial_vol_origin = torch.Tensor([-1., -1., -1.])
        self.white_back = True

    def read_meta(self, near=2.0, far=6.0):


        self.ref_img_idx = self.train_img_idx[0]
        ref_c2w = np.array(self.meta['frames'][self.ref_img_idx]['transform_matrix']) @ self.blender2opencv
        # ref_c2w = torch.FloatTensor(ref_c2w)
        self.ref_c2w = ref_c2w
        self.ref_w2c = np.linalg.inv(ref_c2w)


        w, h = self.img_wh
        self.focal = 0.5 * 800 / np.tan(0.5 * self.meta['camera_angle_x'])  # original focal length
        self.focal *= self.img_wh[0] / 800  # modify focal length to match size self.img_wh

        # bounds, common for all scenes
        self.near = near
        self.far = far
        self.bounds = np.array([self.near, self.far])

        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = get_ray_directions(h, w, [self.focal,self.focal])  # (h, w, 3)
        intrinsics = np.eye(4)
        intrinsics[:3, :3] = np.array([[self.focal,0,w/2],[0,self.focal,h/2],[0,0,1]]).astype(np.float32)
        self.intrinsics = intrinsics

        self.image_paths = []
        self.poses = []
        self.all_rays = []
        self.all_images = []
        self.all_masks = []
        self.all_w2cs = []
        self.all_intrinsics = []
        for frame in self.meta['frames']:
            pose = np.array(frame['transform_matrix']) @ self.blender2opencv
            self.poses += [pose]
            c2w = torch.FloatTensor(pose)
            w2c = np.linalg.inv(c2w)
            image_path = os.path.join(self.root_dir, f"{frame['file_path']}.png")
            self.image_paths += [image_path]
            img = Image.open(image_path)
            img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img)  # (4, h, w)
            
            self.all_masks += [img[-1:,:]>0]
            # img = img[:3, :] * img[ -1:,:] + (1 - img[-1:, :])  # blend A to RGB
            img = img[:3, :] * img[ -1:,:] 
            img = img.numpy()  # (3, h, w)
            self.all_images += [img]


            self.all_masks += []
            self.all_intrinsics.append(self.intrinsics)
            # - transform from world system to ref-camera system
            self.all_w2cs.append(w2c @ np.linalg.inv(self.ref_w2c))
            
        self.all_images = torch.from_numpy(np.stack(self.all_images)).to(torch.float32)
        self.all_intrinsics = torch.from_numpy(np.stack(self.all_intrinsics)).to(torch.float32)
        self.all_w2cs = torch.from_numpy(np.stack(self.all_w2cs)).to(torch.float32)
        # self.img_wh = [self.img_wh[0] - self.clip_wh[0] - self.clip_wh[2],
        #                self.img_wh[1] - self.clip_wh[1] - self.clip_wh[3]]

    def cal_scale_mat(self, img_hw, intrinsics, extrinsics, near_fars, factor=1.):
        center, radius, _ = get_boundingbox(img_hw, intrinsics, extrinsics, near_fars)
        radius = radius * factor
        scale_mat = np.diag([radius, radius, radius, 1.0])
        scale_mat[:3, 3] = center.cpu().numpy()
        scale_mat = scale_mat.astype(np.float32)

        return scale_mat, 1. / radius.cpu().numpy()

    def scale_cam_info(self):
        new_intrinsics = []
        new_near_fars = []
        new_w2cs = []
        new_c2ws = []
        new_affine_mats = []
        for idx in range(len(self.all_images)):

            intrinsics = self.all_intrinsics[idx]
            # import ipdb; ipdb.set_trace()
            P = intrinsics @ self.all_w2cs[idx] @ self.scale_mat
            P = P.cpu().numpy()[:3, :4]

            # - should use load_K_Rt_from_P() to obtain c2w
            c2w = load_K_Rt_from_P(None, P)[1]
            w2c = np.linalg.inv(c2w)
            new_w2cs.append(w2c)
            new_c2ws.append(c2w)
            new_intrinsics.append(intrinsics)
            affine_mat = np.eye(4)
            affine_mat[:3, :4] = intrinsics[:3, :3] @ w2c[:3, :4]
            new_affine_mats.append(affine_mat)

            camera_o = c2w[:3, 3]
            dist = np.sqrt(np.sum(camera_o ** 2))
            near = dist - 1
            far = dist + 1

            new_near_fars.append([0.95 * near, 1.05 * far])

        new_intrinsics, new_w2cs, new_c2ws, new_affine_mats, new_near_fars = \
            np.stack(new_intrinsics), np.stack(new_w2cs), np.stack(new_c2ws), \
            np.stack(new_affine_mats), np.stack(new_near_fars)

        new_intrinsics = torch.from_numpy(np.float32(new_intrinsics))
        new_w2cs = torch.from_numpy(np.float32(new_w2cs))
        new_c2ws = torch.from_numpy(np.float32(new_c2ws))
        new_affine_mats = torch.from_numpy(np.float32(new_affine_mats))
        new_near_fars = torch.from_numpy(np.float32(new_near_fars))

        return new_intrinsics, new_w2cs, new_c2ws, new_affine_mats, new_near_fars

    def load_poses_all(self, file=f"transforms_train.json"):
        with open(os.path.join(self.root_dir, file), 'r') as f:
            meta = json.load(f)

        c2ws = []
        for i,frame in enumerate(meta['frames']):
            c2ws.append(np.array(frame['transform_matrix']) @ self.blender2opencv)
        return np.stack(c2ws)

    def define_transforms(self):
        self.transform = T.ToTensor()



    def get_conditional_sample(self):
        sample = {}
        support_idxs = self.train_img_idx

        sample['images'] = self.all_images[support_idxs]  # (V, 3, H, W)
        sample['w2cs'] = self.scaled_w2cs[self.train_img_idx]  # (V, 4, 4)
        sample['c2ws'] = self.scaled_c2ws[self.train_img_idx]  # (V, 4, 4)
        sample['near_fars'] = self.scaled_near_fars[self.train_img_idx]  # (V, 2)
        sample['intrinsics'] = self.scaled_intrinsics[self.train_img_idx][:, :3, :3]  # (V, 3, 3)
        sample['affine_mats'] = self.scaled_affine_mats[self.train_img_idx]  # ! in world space

        # sample['scan'] = self.scan_id
        sample['scale_factor'] = torch.tensor(self.scale_factor)
        sample['scale_mat'] = torch.from_numpy(self.scale_mat)
        sample['trans_mat'] = torch.from_numpy(np.linalg.inv(self.ref_w2c))
        sample['img_wh'] = torch.from_numpy(np.array(self.img_wh))
        sample['partial_vol_origin'] = torch.tensor(self.partial_vol_origin, dtype=torch.float32)

        return sample



    def __len__(self):
        if self.split == 'train':
            return self.n_views * 1000
        else:
            return len(self.test_img_idx) * 1000


    def __getitem__(self, idx):
        sample = {}

        if self.split == 'train':
            render_idx = self.train_img_idx[idx % self.n_views]
            support_idxs = [idx for idx in self.train_img_idx if idx != render_idx]
        else:
            # render_idx = idx % self.n_test_images + self.n_train_images
            render_idx = self.test_img_idx[idx % len(self.test_img_idx)]
            support_idxs = [render_idx]

        sample['images'] = self.all_images[support_idxs]  # (V, 3, H, W)
        sample['w2cs'] = self.scaled_w2cs[support_idxs]  # (V, 4, 4)
        sample['c2ws'] = self.scaled_c2ws[support_idxs]  # (V, 4, 4)
        sample['intrinsics'] = self.scaled_intrinsics[support_idxs][:, :3, :3]  # (V, 3, 3)
        sample['affine_mats'] = self.scaled_affine_mats[support_idxs]  # ! in world space
        # sample['scan'] = self.scan_id
        sample['scale_factor'] = torch.tensor(self.scale_factor)
        sample['img_wh'] = torch.from_numpy(np.array(self.img_wh))
        sample['partial_vol_origin'] = torch.tensor(self.partial_vol_origin, dtype=torch.float32)
        sample['img_index'] = torch.tensor(render_idx)

        # - query image
        sample['query_image'] = self.all_images[render_idx]
        sample['query_c2w'] = self.scaled_c2ws[render_idx]
        sample['query_w2c'] = self.scaled_w2cs[render_idx]
        sample['query_intrinsic'] = self.scaled_intrinsics[render_idx]
        sample['query_near_far'] = self.scaled_near_fars[render_idx]
        # sample['meta'] = str(self.scan_id) + "_" + os.path.basename(self.images_list[render_idx])
        sample['scale_mat'] = torch.from_numpy(self.scale_mat)
        sample['trans_mat'] = torch.from_numpy(np.linalg.inv(self.ref_w2c))
        sample['rendering_c2ws'] = self.scaled_c2ws[self.test_img_idx]
        sample['rendering_imgs_idx'] = torch.Tensor(np.array(self.test_img_idx).astype(np.int32))

        # - generate rays
        if self.split == 'val' or self.split == 'test':
            sample_rays = gen_rays_from_single_image(
                self.img_wh[1], self.img_wh[0],
                sample['query_image'],
                sample['query_intrinsic'],
                sample['query_c2w'],
                depth=None,
                mask=None)
        else:
            sample_rays = gen_random_rays_from_single_image(
                self.img_wh[1], self.img_wh[0],
                self.N_rays,
                sample['query_image'],
                sample['query_intrinsic'],
                sample['query_c2w'],
                depth=None,
                mask=None,
                dilated_mask=None,
                importance_sample=False,
                h_patch_size=self.h_patch_size
            )

        sample['rays'] = sample_rays

        return sample