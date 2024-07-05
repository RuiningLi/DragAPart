import os
from os.path import join as pjoin
import json
import numpy as np
import time
import random
from argparse import ArgumentParser

from utils.config_utils import (
    PARTNET_DATASET_PATH, AKB48_DATASET_PATH, 
    PARTNET_ID_PATH, AKB48_ID_PATH, 
    PARTNET_CAMERA_POSITION_RANGE, AKB48_CAMERA_POSITION_RANGE, 
    TARGET_GAPARTS, TARGET_MOVING_PARTS, BACKGROUND_RGB, SAVE_PATH
)
from utils.read_utils import (
    get_id_category, read_joints_from_urdf_file, 
    save_rgb_image, save_depth_map, save_anno_dict, save_meta
)
from utils.render_utils import (
    get_cam_pos, set_all_scene, render_rgba_image, render_depth_map,
    render_sem_ins_seg_map, add_background_color_for_image, 
    get_camera_pos_mat, merge_joint_qpos, update_pos
)
from utils.pose_utils import (
    query_part_pose_from_joint_qpos, 
    get_NPCS_map_from_oriented_bbox
)


def render_one(
    dataset_name, 
    model_id, 
    camera_idx, 
    render_idx, 
    height, 
    width, 
    use_raytracing=False, 
    replace_texture=False, 
    num_frames=1, 
    randomize_static_part_pose=False, 
    randomize_speed=False
):

    np.random.seed((os.getpid() * int(time.time())) % 123456789)
    random.seed((os.getpid() * int(time.time())) % 123456789)

    try:
    
    # 1. read the id list to get the category; set path, camera range, and base link name
        if dataset_name == 'partnet':
            category = get_id_category(model_id, PARTNET_ID_PATH)
            if category is None:
                raise ValueError(f'Cannot find the category of model {model_id}')
            data_path = pjoin(PARTNET_DATASET_PATH, str(model_id))
            camera_position_range = PARTNET_CAMERA_POSITION_RANGE
            base_link_name = 'base'
            
        elif dataset_name == 'akb48':
            category = get_id_category(model_id, AKB48_ID_PATH)
            if category is None:
                raise ValueError(f'Cannot find the category of model {model_id}')
            data_path = pjoin(AKB48_DATASET_PATH, category, str(model_id))
            camera_position_range = AKB48_CAMERA_POSITION_RANGE
            base_link_name = 'root'
        
        else:
            raise ValueError(f'Unknown dataset {dataset_name}')
        
        # 2. read the urdf file,  get the kinematic chain, and collect all the joints information
        joints_dict = read_joints_from_urdf_file(data_path, 'mobility_annotation_gapartnet.urdf')
        child_link_to_joint_name = {}
        for joint_name, joint_dict in joints_dict.items():
            child_link_to_joint_name[joint_dict['child']] = joint_name
        
        # 3. generate the joint qpos randomly in the limit range
        anno_path = pjoin(data_path, "link_annotation_gapartnet.json")
        anno_list = json.load(open(anno_path, 'r'))
        movable_joint_names = [
            child_link_to_joint_name[anno["link_name"]] 
            for anno in anno_list if anno["is_gapart"] and anno["category"] in TARGET_MOVING_PARTS
        ]

        # Make sure the joints don't have hiararchy
        for joint_name in movable_joint_names:
            parent_link = joints_dict[joint_name]['parent']
            while parent_link != base_link_name:
                next_joint = child_link_to_joint_name[parent_link]
                assert next_joint not in movable_joint_names, \
                    f'Joint {joint_name} is not movable because its parent {parent_link} is movable.'
                parent_link = joints_dict[next_joint]['parent']

        # Save the moving links
        joint_to_moving_links = {}
        for link_name in child_link_to_joint_name:
            joint = child_link_to_joint_name[link_name]
            if joint in movable_joint_names:
                joint_to_moving_links[joint] = joint_to_moving_links.get(joint, []) + [link_name]
                continue
            parent_link = joints_dict[joint]['parent']
            while parent_link != base_link_name:
                next_joint = child_link_to_joint_name[parent_link]
                if next_joint in movable_joint_names:
                    joint_to_moving_links[next_joint] = joint_to_moving_links.get(next_joint, []) + [link_name]
                    break
                parent_link = joints_dict[next_joint]['parent']
        
        for joint_name in movable_joint_names:
            assert joint_name in joints_dict and joints_dict[joint_name]['type'] != 'fixed', f'Joint {joint_name} is not movable.'

        if len(movable_joint_names) <= 3:
            # Every single subset of movable joints is a combination
            moving_combinations = [[joint_name] for joint_name in movable_joint_names] + \
                [[movable_joint_names[i], movable_joint_names[j]] for i in range(len(movable_joint_names)) for j in range(i+1, len(movable_joint_names))] + \
                ([movable_joint_names] if len(movable_joint_names) == 3 else [])
        else:
            # Every subset of size 1 is a combination, randomly select 1 subsets of size 2, 3, ..., len(movable_joint_names)
            moving_combinations = [[joint_name] for joint_name in movable_joint_names] + \
                [list(
                    np.random.choice(movable_joint_names, num_moving, replace=False)
                ) for num_moving in range(2, len(movable_joint_names) + 1)]
        
        if len(movable_joint_names) == 0:
            return
        
        for cid, moving_combination in enumerate(moving_combinations):
                
            # 4. generate the camera pose randomly in the specified range
            camera_range = camera_position_range[category][camera_idx]
            camera_pos = get_cam_pos(
                theta_min=camera_range['theta_min'], theta_max=camera_range['theta_max'],
                phi_min=camera_range['phi_min'], phi_max=camera_range['phi_max'],
                dis_min=camera_range['distance_min'], dis_max=camera_range['distance_max']
            )
            fov = np.random.uniform(30, 45)

            moving_links_name = []
            for joint_name in moving_combination:
                moving_links_name += joint_to_moving_links[joint_name]

            static_joint_qpos = {}
            for joint_name in joints_dict:
                if joint_name not in moving_combination:
                    joint_type = joints_dict[joint_name]['type']
                    if joint_type == "fixed":
                        static_joint_qpos[joint_name] = 0.0
                    elif joint_type == 'prismatic' or joint_type == 'revolute':
                        joint_limit = joints_dict[joint_name]['limit']
                        static_joint_qpos[joint_name] = np.random.uniform(joint_limit[0], joint_limit[1]) if randomize_static_part_pose else joint_limit[0]
                    elif joint_type == 'continuous':
                        static_joint_qpos[joint_name] = np.random.uniform(-10000.0, 10000.0)
                    
            moving_joint_speed, moving_joint_start_pos = {}, {}
            for joint_name in joints_dict:
                joint_type = joints_dict[joint_name]['type']
                if joint_name in moving_combination:
                    if joint_type == 'prismatic' or joint_type == 'revolute':
                        joint_limit = joints_dict[joint_name]['limit']
                        moving_joint_speed[joint_name] = np.random.uniform(
                            (joint_limit[1] - joint_limit[0]) / num_frames / 2, 
                            (joint_limit[1] - joint_limit[0]) / num_frames
                        ) if randomize_speed else (joint_limit[1] - joint_limit[0]) / (num_frames - 1)
                        moving_joint_start_pos[joint_name] = np.random.uniform(
                            joint_limit[0], 
                            joint_limit[1] - moving_joint_speed[joint_name] * (num_frames - 1)
                        ) if randomize_speed else joint_limit[0]
                    # elif joint_type == 'continuous':
                    #     moving_joint_speed[joint_name] = 0  # ignore the continuous joint
                    else:
                        raise ValueError(f'Unknown joint type {joint_type}')
                
            # 5. pass the joint qpos and the augmentation parameters to set up render environment and robot
            scene, camera, engine, robot = set_all_scene(
                data_path=data_path, 
                urdf_file='mobility_annotation_gapartnet.urdf',
                cam_pos=camera_pos,
                fov=fov,
                width=width,
                height=height,
                use_raytracing=False,
                joint_qpos_dict={**static_joint_qpos, **moving_joint_start_pos},
            )
            
            save_root = pjoin(SAVE_PATH, category, str(model_id), f"{cid:03d}_total{len(moving_combination):02d}")
            os.makedirs(save_root, exist_ok=True)
                        
            for t in range(num_frames):
                moving_joint_qpos = {}
                for joint_name in joints_dict:
                    joint_type = joints_dict[joint_name]['type']
                    if joint_name in moving_combination:
                        if joint_type == 'prismatic' or joint_type == 'revolute':
                            joint_limit = joints_dict[joint_name]['limit']
                            moving_joint_qpos[joint_name] = moving_joint_start_pos[joint_name] + moving_joint_speed[joint_name] * t
                        # elif joint_type == 'continuous':
                        #     moving_joint_qpos[joint_name] = 0  # ignore the continuous joint
                        else:
                            raise ValueError(f'Unknown joint type {joint_type}')
                joint_qpos = {**static_joint_qpos, **moving_joint_qpos}

                robot, scene, camera = update_pos(robot, joint_qpos, scene, camera)
                
                # 6. use qpos to calculate the gapart poses
                link_pose_dict = query_part_pose_from_joint_qpos(
                    data_path=data_path, 
                    anno_file='link_annotation_gapartnet.json', 
                    joint_qpos=joint_qpos, 
                    joints_dict=joints_dict, 
                    target_parts=TARGET_GAPARTS, 
                    base_link_name=base_link_name, 
                    robot=robot
                )
                
                # 7. render the rgb, depth, mask, valid (visible) gaparts
                rgba_image = render_rgba_image(camera=camera)
                rgb_image = rgba_image[:, :, :3]
                depth_map = render_depth_map(camera=camera)
                sem_seg_map, ins_seg_map, valid_linkName_to_instId = render_sem_ins_seg_map(
                    scene=scene, 
                    camera=camera, 
                    link_pose_dict=link_pose_dict, 
                    depth_map=depth_map
                )
                linkName_to_instId = valid_linkName_to_instId
                for linkName in link_pose_dict:
                    if linkName not in linkName_to_instId:
                        linkName_to_instId[linkName] = -1
                joint_to_moving_links_new = {}
                for joint_name in joint_to_moving_links.keys():
                    joint_to_moving_links_new[joint_name] = [
                        valid_linkName_to_instId[link_name] 
                        for link_name in joint_to_moving_links[joint_name] 
                        if link_name in valid_linkName_to_instId
                    ]

                valid_link_pose_dict = {link_name: link_pose_dict[link_name] for link_name in valid_linkName_to_instId.keys()}
                
                # 8. acquire camera intrinsic and extrinsic matrix
                camera_intrinsic, world2camera_rotation, camera2world_translation = get_camera_pos_mat(camera)
                
                # 9. calculate NPCS map
                linkPose_RTS_dict, valid_NPCS_map = get_NPCS_map_from_oriented_bbox(depth_map, ins_seg_map, linkName_to_instId, link_pose_dict, camera_intrinsic, world2camera_rotation, camera2world_translation)
                
                # 10. (optional, only for [partnet] dataset) use texture to render rgb to replace the previous rgb (texture issue during cutting the mesh)
                if replace_texture:
                    assert dataset_name == 'partnet', 'Texture replacement is only needed for PartNet dataset'
                    texture_joints_dict = read_joints_from_urdf_file(data_path, 'mobility_texture_gapartnet.urdf')
                    texture_joint_qpos = merge_joint_qpos(joint_qpos, joints_dict, texture_joints_dict)
                    scene, camera, engine, robot = set_all_scene(data_path=data_path,
                                                    urdf_file='mobility_texture_gapartnet.urdf',
                                                    cam_pos=camera_pos,
                                                    fov=fov,
                                                    width=width,
                                                    height=height,
                                                    use_raytracing=use_raytracing,
                                                    joint_qpos_dict=texture_joint_qpos,
                                                    engine=engine)
                    rgb_image = render_rgba_image(camera=camera)

                rgb_image = add_background_color_for_image(rgb_image, depth_map, BACKGROUND_RGB)
                
                # 11. save the results.
                save_name = f"c{render_idx:03d}_t{t:03d}"
                save_rgb_image(rgba_image, save_root, save_name)
                save_depth_map(depth_map, save_root, save_name)
                
                bbox_pose_dict = {}
                for link_name in valid_link_pose_dict:
                    bbox_pose_dict[link_name] = {
                        'bbox': valid_link_pose_dict[link_name]['bbox'],
                        'category_id': valid_link_pose_dict[link_name]['category_id'],
                        'instance_id': valid_linkName_to_instId[link_name],
                        'pose_RTS_param': linkPose_RTS_dict[link_name],
                        'link_name': link_name,
                    }
                for link_name in link_pose_dict:
                    if link_name not in bbox_pose_dict:
                        bbox_pose_dict[link_name] = {
                            'instance_id': -1,
                            'pose_RTS_param': linkPose_RTS_dict[link_name],
                            'link_name': link_name,
                        }
                anno_dict = {
                    'semantic_segmentation': sem_seg_map,
                    'instance_segmentation': ins_seg_map,
                    'npcs_map': valid_NPCS_map,
                    'bbox_pose_dict': bbox_pose_dict,
                }
                save_anno_dict(anno_dict, save_root, save_name)
                
                metafile = {
                    'model_id': model_id,
                    'category': category,
                    'camera_idx': camera_idx,
                    'render_idx': render_idx,
                    'width': width,
                    'height': height,
                    'joint_qpos': joint_qpos,
                    'camera_pos': camera_pos.reshape(-1).tolist(),
                    'camera_intrinsic': camera_intrinsic.reshape(-1).tolist(),
                    'world2camera_rotation': world2camera_rotation.reshape(-1).tolist(),
                    'camera2world_translation': camera2world_translation.reshape(-1).tolist(),
                    'target_gaparts': TARGET_GAPARTS,
                    'use_raytracing': use_raytracing,
                    'replace_texture': replace_texture,
                }
                for joint_name in moving_combination:
                    metafile[f'{joint_name}_moving_links'] = joint_to_moving_links[joint_name]
                save_meta(metafile, save_root, save_name)
                
                print(f"Rendered {save_name} successfully!")
        
    except Exception as e:
        print(f"Failed to render {model_id} : {e}")
        return
    

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default='partnet', help='Specify the dataset to render')
    parser.add_argument('--model_id', type=int, default=41083, help='Specify the model id to render')
    parser.add_argument('--camera_idx', type=int, default=0, help='Specify the camera range index to render')
    parser.add_argument('--render_idx', type=int, default=0, help='Specify the render index to render')
    parser.add_argument('--height', type=int, default=800, help='Specify the height of the rendered image')
    parser.add_argument('--width', type=int, default=800, help='Specify the width of the rendered image')
    parser.add_argument('--ray_tracing', type=bool, default=False, help='Specify whether to use ray tracing in rendering')
    parser.add_argument('--replace_texture', type=bool, default=False, help='Specify whether to replace the texture of the rendered image using the original model')
    parser.add_argument('--num_frames', type=int, default=36, help='Specify the number of frames to render')
    parser.add_argument('--randomize_static_part_pose', action="store_true", help='Specify whether to randomize the pose of static parts')
    parser.add_argument('--randomize_speed', action="store_true", help='Specify whether to randomize the speed of moving parts')
    
    args = parser.parse_args()
    
    assert args.dataset in ['partnet', 'akb48'], f'Unknown dataset {args.dataset}'
    if args.dataset == 'akb48':
        assert not args.replace_texture, 'Texture replacement is not needed for AKB48 dataset'
    
    render_one(args.dataset, args.model_id, args.camera_idx, args.render_idx, args.height, args.width, args.ray_tracing, args.replace_texture, args.num_frames, args.randomize_static_part_pose, args.randomize_speed)
    
    print("Done!")
