import json
import os
import os.path as osp
import pickle
import random
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import CLIPProcessor


class DragAMoveDataset(Dataset):
    """
    Dataset to preprocess the raw data from the Drag-a-Move dataset.

    Dataset keys:
    - recon_pixel_values: batched pixel values of the image to be reconstructed of shape (B, 3, sample_size, sample_size) of range [-1, 1].
    - cond_pixel_values: batched pixel values of the conditioning image of shape (B, 3, sample_size, sample_size) of range [-1, 1].
    - category: length-B list of strings to indicate the object categories of the batch.
    - drags: batched drag values of shape (B, max_num_drags, 4) where the last dimension is (x, y, u, v) in the range [0, 1],
            where (x, y) is the starting point and (u, v) is the ending point of the drag. If there are less than max_num_drags,
            drags, the rest of the values are zeros.
    """
    def __init__(
        self,
        needed_instance_idx: Optional[List[str]] = None,
        discarded_instance_idx: Optional[List[str]] = None,
        dataset_root_folder: Optional[str] = None,
        sample_size: int = 256,
        total_num_frames: int = 36,
        total_num_cam_viewpoints: int = 32,
        category: str = "ALL",
        background: Tuple[int] = (1, 1, 1),
        fix_motion_time: int = -1,
        fix_cam_viewpoints: int = -1,
        fix_t_cond: int = -1,
        max_num_drags: int = 10,
        fix_num_moving_parts: int = -1,
        enable_horizontal_flip: bool = True,
        random_category_prob: float = 0.5,
        only_large_motion: bool = False,
        extra_keys: Optional[List[str]] = None,
        verbose: bool = False,
    ):
        """
        Args:
        - needed_instance_idx: list of instance indices to be included in the dataset.
        - discarded_instance_idx: list of instance indices to be excluded in the dataset.
        - dataset_root_folder: root folder of the dataset.
        - sample_size: size of the image.
        - total_num_frames: total number of frames in the dataset.
        - total_num_cam_viewpoints: total number of camera viewpoints in the dataset.
        - category: list of categories to be included in the dataset, or "ALL" if all are needed.
        - background: background color of the image, range [-1, 1] for each of the R, G, B channels.
        - fix_motion_time: fixed motion time to be used for all instances, -1 for random selection.
        - fix_cam_viewpoints: fixed camera viewpoint to be used for all instances, -1 for random selection.
        - fix_t_cond: fixed motion time for the conditioning image, -1 for random selection.
        - max_num_drags: maximum number of drags to be sampled for each instance.
        - fix_num_moving_parts: fixed number of moving parts to be used for all instances, -1 for random selection.
        - enable_horizontal_flip: whether to enable horizontal flip augmentation.
        - random_category_prob: probability of selecting a category uniformly at random; 
            the rest (1 - random_category_prob) is weighted by the number of instances in the category.
        - only_large_motion: whether to sample only large motions, i.e., conditioning image and reconstruction image are
            guaranteed to have large time difference.
        - extra_keys: list of extra keys to be included in the dataset. Currently support: clip_pixel_values.
        - verbose: whether to print verbose information.
        """
        super().__init__()
        self.sample_size = (sample_size, sample_size)

        self.dataset_root_folder = dataset_root_folder
        self.only_large_motion = only_large_motion

        if category == "ALL":
            self.categories = list(os.listdir(self.dataset_root_folder))
        else:
            self.categories = [category]

        self.fix_t = fix_motion_time
        self.fix_c = fix_cam_viewpoints
        self.fix_t_cond = fix_t_cond
        self.fix_num_moving_parts = fix_num_moving_parts

        category_instance_tuples = []
        category_to_instances = {}
        if discarded_instance_idx is None:
            discarded_instance_idx = []

        for c in self.categories:
            if needed_instance_idx is None:
                category_instance_tuples += [
                    (c, idx)
                    for idx in os.listdir(osp.join(self.dataset_root_folder, c))
                    if idx not in discarded_instance_idx
                ]
            else:
                category_instance_tuples += [
                    (c, idx)
                    for idx in os.listdir(osp.join(self.dataset_root_folder, c))
                    if (
                        idx in needed_instance_idx and idx not in discarded_instance_idx
                    )
                ]
        for c, idx in category_instance_tuples:
            if c not in category_to_instances:
                category_to_instances[c] = []
            category_to_instances[c].append(idx)

        self.categories = [c for c in self.categories if c in category_to_instances]
        self.category_to_instances = category_to_instances

        self.total_num_instances = len(category_instance_tuples)
        # self.category_to_instances = category_to_instances
        self.category_sample_probs = [
            random_category_prob / len(self.categories) + (1 - random_category_prob) * len(category_to_instances[c]) / self.total_num_instances for c in self.categories
        ]

        self.pixel_transforms = transforms.Compose([
            transforms.Resize(self.sample_size, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])

        self.total_num_frames = total_num_frames
        self.background = background
        self.total_num_cam_viewpoints = total_num_cam_viewpoints
        self.enable_horizontal_flip = enable_horizontal_flip
        self.max_num_drags = max_num_drags

        self.extra_keys = extra_keys
        if extra_keys is not None:
            for key in extra_keys:
                if key == "clip_pixel_values":
                    self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

        self.verbose = verbose

    def _add_background(self, pixel_values: torch.Tensor):
        assert (
            pixel_values.shape[0] == 4
        ), f"pixel_values should have 4 channels, but got {pixel_values.shape[1]}"
        pixel_values[:3] = (
            pixel_values[:3] * (pixel_values[3:4] > 0).float()
            + torch.Tensor(self.background)
            .to(pixel_values.device)[:, None, None]
            .repeat(1, *pixel_values.shape[1:])
            * (pixel_values[3:4] <= 0).float()
        )
        return pixel_values

    def sample_drags(
        self,
        pose_idx,
        recon_motion_t,
        cond_motion_t,
        root,
    ):
        seg_root = osp.join(root, "segmentation")
        seg = np.load(osp.join(seg_root, f"c{pose_idx:03d}_t{cond_motion_t:03d}.npz"))
        instance_segmentation = seg["instance_segmentation"]
        meta_root = osp.join(root, "metafile")
        meta_cond = json.load(open(osp.join(meta_root, f"c{pose_idx:03d}_t{cond_motion_t:03d}.json")))
        meta_recon = json.load(open(osp.join(meta_root, f"c{pose_idx:03d}_t{recon_motion_t:03d}.json")))
        npcs_root = osp.join(root, "npcs")
        npcs = np.load(osp.join(npcs_root, f"c{pose_idx:03d}_t{cond_motion_t:03d}.npz"))
        npcs_map = npcs["npcs_map"]
        bbox_root = osp.join(root, "bbox")
        bbox_dict = pickle.load(open(osp.join(bbox_root, f"c{pose_idx:03d}_t{recon_motion_t:03d}.pkl"), "rb"))
        bbox_pose_dict = bbox_dict['bbox_pose_dict']

        w2c_R = np.array(meta_recon["world2camera_rotation"]).reshape(3, 3)
        c2w_T = np.array(meta_recon["camera2world_translation"]).reshape(3, 1)
        K = np.array(meta_recon["camera_intrinsic"]).reshape(3, 3)

        flow_dense = torch.zeros(2, *instance_segmentation.shape)
        drags = torch.zeros(self.max_num_drags, 4)
        
        all_moving_link_names = []
        for k, v in meta_cond.items():
            if k.endswith("moving_links"):
                all_moving_link_names += v
        all_moving_instance_ids = [bbox_pose_dict[link_name]["instance_id"] for link_name in all_moving_link_names]
        
        moving_instance_ids = []
        for k, v in meta_cond.items():
            if k.endswith("moving_links"):
                ids = [bbox_pose_dict[link_name]["instance_id"] for link_name in v]
                valid_v = [vv for vv in ids if (instance_segmentation == vv).sum() >= 1 and vv >= 0]
                if len(valid_v) > 0: moving_instance_ids.append(random.choice(valid_v))
        
        for moving_instance_link_name, moving_instance_id in zip(all_moving_link_names, all_moving_instance_ids):
            if moving_instance_id < 0: continue
            y, x = np.where(instance_segmentation == moving_instance_id)
            
            canon_pos = npcs_map[y, x]
            pose_RTS_param = bbox_pose_dict[moving_instance_link_name]["pose_RTS_param"]
            R = pose_RTS_param["R"]
            T = pose_RTS_param["T"]
            scaler = pose_RTS_param["scaler"]

            world_pos = canon_pos @ np.linalg.inv(R.T) * scaler + T
            camera_pos = (world_pos - c2w_T.T) @ w2c_R
            camera_x, camera_y, camera_z = camera_pos[:, 0], camera_pos[:, 1], camera_pos[:, 2]
            u = camera_x * K[0, 0] / camera_z + K[0, 2]
            v = camera_y * K[1, 1] / camera_z + K[1, 2]

            flow_dense[:, y, x] = torch.stack([torch.Tensor(u), torch.Tensor(v)], dim=0) - \
                torch.stack([torch.Tensor(x), torch.Tensor(y)], dim=0)

        current_drag_id = 0
        for moving_instance_id in moving_instance_ids:
            y, x = np.where(instance_segmentation == moving_instance_id)
            flow_magnitude = flow_dense[:, y, x].norm(dim=0)
            if flow_magnitude.sum() > 0:
                random_idx = np.random.choice(list(range(len(y))), size=1, p=(flow_magnitude / flow_magnitude.sum()).numpy())
                y, x = y[random_idx], x[random_idx]
                u, v = x[0] + flow_dense[0, y, x].item(), y[0] + flow_dense[1, y, x].item()
                if current_drag_id < self.max_num_drags:
                    drags[current_drag_id, :] = torch.Tensor([x[0] / 512., y[0] / 512., u / 512., v / 512.])  # The images are rendered with 512x512 resolution.
                current_drag_id += 1

        return drags

    def get_batch(self, idx):
            try:
                category = random.choices(self.categories, weights=self.category_sample_probs, k=1)[0]
                instances = self.category_to_instances[category]
                instance_idx = random.choice(instances)

                instance_root = osp.join(self.dataset_root_folder, category, instance_idx)

                all_subsets = os.listdir(instance_root)
                if self.fix_num_moving_parts > 0:
                    all_subsets = [s for s in all_subsets if int(s.split("total")[-1]) == self.fix_num_moving_parts]
                subset_idx = np.random.choice(all_subsets)
                root = osp.join(instance_root, subset_idx)

                cond_motion_t_to_select = list(range(self.total_num_frames))
                if self.only_large_motion:
                    relative_t = np.random.randint(int(self.total_num_frames * 0.5), self.total_num_frames)
                    if np.random.rand() < 0.5:
                        relative_t = -relative_t
                else:
                    relative_t = np.random.randint(-self.total_num_frames + 1, self.total_num_frames)
                                        
                cond_motion_t_to_select = [
                    t for t in cond_motion_t_to_select if t + relative_t >= 0 and t + relative_t < self.total_num_frames
                ]
                cond_motion_t = (
                    (
                        np.random.choice(cond_motion_t_to_select)
                    )
                    if self.fix_t < 0
                    else self.fix_t
                )
                recon_motion_t = (
                    cond_motion_t + relative_t
                    if self.fix_t < 0
                    else self.fix_t
                )

                recon_pose_idx = (
                    np.random.randint(0, self.total_num_cam_viewpoints) if self.fix_c < 0 else self.fix_c
                )
                cond_pose_idx = recon_pose_idx
                pose_idx = recon_pose_idx
                
                drags = self.sample_drags(pose_idx, recon_motion_t, cond_motion_t, root)

                instance_root = osp.join(root, "rgb")
                recon_rgb_path = osp.join(instance_root, f"c{recon_pose_idx:03d}_t{recon_motion_t:03d}.png")
                cond_rgb_path = osp.join(instance_root, f"c{cond_pose_idx:03d}_t{cond_motion_t:03d}.png")

                recon_pixel_values = self.pixel_transforms(Image.open(recon_rgb_path).convert("RGB"))
                cond_image = Image.open(cond_rgb_path).convert("RGB")
                cond_pixel_values = self.pixel_transforms(cond_image)

                extra_values = {}
                if self.extra_keys is not None:
                    for key in self.extra_keys:
                        if key == "clip_pixel_values":
                            extra_values["clip_pixel_values"] = self.clip_processor(
                                images=cond_image, return_tensors="pt"
                            ).pixel_values
                        else:
                            raise NotImplementedError(f"Extra key {key} is not supported.")

            except Exception as e:
                if self.verbose: print(e)
                return None
            
            horizontal_flip = random.random() < 0.5
            if self.enable_horizontal_flip and horizontal_flip:
                recon_pixel_values = torch.flip(recon_pixel_values, dims=[-1])
                cond_pixel_values = torch.flip(cond_pixel_values, dims=[-1])
                not_all_zeros = drags.any(dim=1)
                drags[not_all_zeros, 0] = 1 - drags[not_all_zeros, 0]
                drags[not_all_zeros, 2] = 1 - drags[not_all_zeros, 2]

            datum = dict(
                recon_pixel_values=recon_pixel_values,
                cond_pixel_values=cond_pixel_values,
                category=category,
                drags=drags,
            )
            datum.update(extra_values)
            return datum

    def __len__(self):
        return self.total_num_instances

    def __getitem__(self, idx):
        while True:
            batch = self.get_batch(idx)
            if batch is not None:
                return batch
            idx = random.randint(0, self.total_num_instances - 1)


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    visualize_dir = "./assets/dataset_visualization"
    os.makedirs(visualize_dir, exist_ok=True)

    random.seed(1024)
    np.random.seed(1024)

    sample_size, num_max_drags = 256, 10
    dataset = DragAMoveDataset(
        dataset_root_folder="/scratch/shared/beegfs/ruining/data/GAPartNet-rendering-v7/train",
        only_large_motion=True,
        sample_size=sample_size,
        max_num_drags=num_max_drags,
        random_category_prob=0.2,
    )
    dataloader = DataLoader(dataset, batch_size=1, num_workers=0)

    for bid, batch in enumerate(dataloader):
        recon_pixel_values = batch["recon_pixel_values"]
        cond_pixel_values = batch["cond_pixel_values"]
        category = batch["category"]
        drags = batch["drags"]

        recon_image = np.ascontiguousarray(((recon_pixel_values[0].permute(1, 2, 0).numpy() + 1) * 127.5).astype(np.uint8))
        cond_image = np.ascontiguousarray(((cond_pixel_values[0].permute(1, 2, 0).numpy() + 1) * 127.5).astype(np.uint8))
        for drag_id in range(10):
            x, y, u, v = drags[0, drag_id].numpy()
            if abs(x) + abs(y) + abs(u) + abs(v) < 1e-6:  # no drag
                continue
            cond_image = cv2.arrowedLine(cond_image, (int(x * sample_size), int(y * sample_size)), (int(u * sample_size), int(v * sample_size)), (0, 255, 0), 2)
        
        cond_image = cv2.putText(cond_image, category[0], (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, lineType=cv2.LINE_AA)
        image_to_save = np.concatenate([cond_image, recon_image], axis=1).astype(np.uint8)
        cv2.imwrite(osp.join(visualize_dir, f"batch_{bid}.png"), image_to_save[:, :, ::-1])

        if bid >= 4:
            break
