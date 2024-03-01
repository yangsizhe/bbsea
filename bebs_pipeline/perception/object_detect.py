import cv2
import numpy as np
import supervision as sv

import torch
import torchvision

# from groundingdino.util.inference import Model
# from segment_anything import sam_model_registry, SamPredictor
from scalingup.utils.core import PointCloud, Env
import sys 
sys.path.append("..") 
from utils import get_link_point_cloud, OBJECT_NAMES, LINK_NAMES_AND_LINK_PATHS


# # load the name of all the objects
# OBJECT_NAMES = []
# with open('bebs_pipeline/config/scene.yaml', 'r') as file:
#     scene_config = yaml.safe_load(file)
# for item in scene_config['env']['assets']:
#     OBJECT_NAMES.append(item['name'])
# for item in scene_config['env']['fixed_objects']:
#     OBJECT_NAMES.append(item['name'])

# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# # GroundingDINO config and checkpoint
# Grounded_Segment_Anything_PATH = "/home/szyang/projects/brain_eye_body_synchronization/detector/Grounded-Segment-Anything"  # "/home/ysz/projects/projects_FM4embodiedAI/detection/Grounded-Segment-Anything"
# GROUNDING_DINO_CONFIG_PATH = f"{Grounded_Segment_Anything_PATH}/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
# GROUNDING_DINO_CHECKPOINT_PATH = f"{Grounded_Segment_Anything_PATH}/groundingdino_swint_ogc.pth"
# # Segment-Anything checkpoint
# SAM_ENCODER_VERSION = "vit_h"
# SAM_CHECKPOINT_PATH = f"{Grounded_Segment_Anything_PATH}/sam_vit_h_4b8939.pth"

MAX_OFFSCREEN_HEIGHT = 640
RGB_DTYPE = np.uint8
DEPTH_DTYPE = np.float32

# # Building GroundingDINO inference model
# grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)

# # Building SAM Model and SAM Predictor
# sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)
# sam.to(device=DEVICE)
# sam_predictor = SamPredictor(sam)

# BOX_THRESHOLD = 0.40
# TEXT_THRESHOLD = 0.25


def get_point_cloud_dict(env):
    pcd_dict = {}
    # TODO detect and segment objects, and then construct point cloud, pcd_dict[label] = point_cloud
    
    for object_name in OBJECT_NAMES:
        pcd_dict[object_name] = get_link_point_cloud(env, object_name+'/|'+object_name+'/'+object_name)
    for object_name in LINK_NAMES_AND_LINK_PATHS:
        pcd_dict[object_name] = get_link_point_cloud(env, LINK_NAMES_AND_LINK_PATHS[object_name])

    return pcd_dict

# # Prompting SAM with detected boxes
# def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
#     sam_predictor.set_image(image)
#     result_masks = []
#     for box in xyxy:
#         masks, scores, logits = sam_predictor.predict(
#             box=box,
#             multimask_output=True
#         )
#         index = np.argmax(scores)
#         result_masks.append(masks[index])
#     return np.array(result_masks)

# def detect_and_segment_object(image: np.ndarray, class_text):
#     image = image[:,:,::-1]  # rgb to bgr
#     detections = grounding_dino_model.predict_with_classes(
#         image=image,
#         classes=[class_text,],
#         box_threshold=BOX_THRESHOLD,
#         text_threshold=BOX_THRESHOLD
#     )
#     best_idx = detections.confidence.argmax()
#     detections.xyxy = detections.xyxy[best_idx:best_idx+1]
#     detections.confidence = detections.confidence[best_idx:best_idx+1]
#     detections.class_id = detections.class_id[best_idx:best_idx+1]
#     detections.mask = segment(
#         sam_predictor=sam_predictor,
#         image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
#         xyxy=detections.xyxy
#     ).squeeze()
#     # box_annotator = sv.BoxAnnotator()
#     # annotated_image = box_annotator.annotate(scene=image.copy(), detections=detections, labels=[class_text,])
#     # cv2.imwrite("images&videos/grounded_sam_annotated_image.jpg", annotated_image)
#     # import pdb;pdb.set_trace()
#     return detections.mask


def get_object_point_cloud_via_mask(point_cloud: PointCloud, mask: np.ndarray):
    seg_mask = mask.reshape((-1,))
    if not seg_mask.any():
        return PointCloud(
            xyz_pts=np.empty((0, 3), dtype=DEPTH_DTYPE),
            rgb_pts=np.empty((0, 3), dtype=RGB_DTYPE),
            segmentation_pts={'no_key': np.ones(seg_mask.sum(), dtype=bool)},
        )
    link_point_cloud = PointCloud(
        xyz_pts=point_cloud.xyz_pts[seg_mask],
        rgb_pts=point_cloud.rgb_pts[seg_mask],
        segmentation_pts={'no_key': np.ones(seg_mask.sum(), dtype=bool)},
    )
    if len(link_point_cloud) == 0:
        return link_point_cloud
    # help remove outliers due to noisy segmentations
    # this is actually quite expensive, and can be improved
    _, ind = link_point_cloud.to_open3d().remove_radius_outlier(
        nb_points=32, radius=0.02
    )
    return PointCloud(
        xyz_pts=link_point_cloud.xyz_pts[ind],
        rgb_pts=link_point_cloud.rgb_pts[ind],
        segmentation_pts={'no_key': np.ones(len(ind), dtype=bool)},
    )


# def get_object_point_cloud(env: Env, class_text: str):
#     res = (
#         (
#             np.array(env.config.obs_dim)
#             / max(env.config.obs_dim)
#             * MAX_OFFSCREEN_HEIGHT
#         )
#         .astype(int)
#         .tolist()
#     )
#     images = env.render(obs_dim=(res[0], res[1]))
#     link_point_clouds = []
#     # for sensor_output in images.values():
#     sensor_output0 = images['front']
#     mask0 = detect_and_segment_object(sensor_output0.rgb, class_text)
#     link_point_clouds.append(get_object_point_cloud_via_mask(sensor_output0.point_cloud, mask0))
#     sensor_output1 = images['top_down']
#     mask1 = detect_and_segment_object(sensor_output1.rgb, class_text)
#     link_point_clouds.append(get_object_point_cloud_via_mask(sensor_output1.point_cloud, mask1))

#     link_point_cloud: PointCloud = sum(
#         link_point_clouds[1:], start=link_point_clouds[0]
#     )
#     return link_point_cloud

def get_object_point_cloud(env: Env, object_name: str):
    if object_name in LINK_NAMES_AND_LINK_PATHS:
        return get_link_point_cloud(env, LINK_NAMES_AND_LINK_PATHS[object_name])
    elif object_name in OBJECT_NAMES:
        return get_link_point_cloud(env, object_name+'/|'+object_name+'/'+object_name)
