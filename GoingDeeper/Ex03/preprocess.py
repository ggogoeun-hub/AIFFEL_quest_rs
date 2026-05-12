"""MPII Human Pose Estimation - 데이터 전처리 모듈

이미지 크롭, 히트맵 생성, Dataset 클래스를 제공합니다.
"""

import os
import json
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


# MPII 키포인트 정의
NUM_JOINTS = 16
JOINT_NAMES = [
    '오른쪽 발목', '오른쪽 무릎', '오른쪽 엉덩이',
    '왼쪽 엉덩이', '왼쪽 무릎', '왼쪽 발목',
    '골반', '가슴(흉부)', '목', '머리 위',
    '오른쪽 손목', '오른쪽 팔꿈치', '오른쪽 어깨',
    '왼쪽 어깨', '왼쪽 팔꿈치', '왼쪽 손목'
]

# 스켈레톤 연결 (시각화용)
SKELETON = [
    (0, 1), (1, 2), (2, 6),
    (3, 4), (4, 5), (3, 6),
    (6, 7), (7, 8), (8, 9),
    (10, 11), (11, 12), (12, 7),
    (13, 14), (14, 15), (13, 7),
]


def parse_one_annotation(ann):
    """Annotation 하나를 파싱하여 필요한 정보를 추출합니다."""
    joints = np.array(ann['joints'])
    joints_vis = np.array(ann['joints_vis'])
    image_path = ann['image']
    center = np.array(ann['center'])
    scale = ann['scale']
    return {
        'joints': joints,
        'joints_vis': joints_vis,
        'image': image_path,
        'center': center,
        'scale': scale,
    }


def filter_fully_visible(annotations, max_samples=None):
    """joints_vis가 모두 1인 annotation만 선택합니다."""
    filtered = []
    seen_images = set()
    for ann in annotations:
        if all(v == 1 for v in ann['joints_vis']):
            img_name = ann['image']
            if img_name not in seen_images:
                seen_images.add(img_name)
                filtered.append(ann)
                if max_samples and len(filtered) >= max_samples:
                    break
    return filtered


def crop_image(img, center, scale, output_size=256):
    """center와 scale을 기반으로 이미지를 크롭하고 리사이즈합니다."""
    h, w = img.shape[:2]
    body_size = scale * 200
    crop_size = body_size * 1.25

    x1 = int(center[0] - crop_size / 2)
    y1 = int(center[1] - crop_size / 2)
    x2 = int(center[0] + crop_size / 2)
    y2 = int(center[1] + crop_size / 2)

    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - w)
    pad_bottom = max(0, y2 - h)

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    cropped = img[y1:y2, x1:x2].copy()

    if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
        cropped = cv2.copyMakeBorder(cropped, pad_top, pad_bottom, pad_left, pad_right,
                                      cv2.BORDER_CONSTANT, value=(0, 0, 0))

    if cropped.shape[0] > 0 and cropped.shape[1] > 0:
        cropped = cv2.resize(cropped, (output_size, output_size))
    else:
        cropped = np.zeros((output_size, output_size, 3), dtype=np.uint8)

    transform_params = {
        'crop_x1': x1 - pad_left,
        'crop_y1': y1 - pad_top,
        'crop_size': crop_size,
        'output_size': output_size,
    }
    return cropped, transform_params


def transform_joint_to_crop(joint, transform_params):
    """원본 이미지 좌표를 크롭 이미지 좌표로 변환합니다."""
    x = (joint[0] - transform_params['crop_x1']) / transform_params['crop_size'] * transform_params['output_size']
    y = (joint[1] - transform_params['crop_y1']) / transform_params['crop_size'] * transform_params['output_size']
    return np.array([x, y])


def generate_heatmaps(joints, joints_vis, transform_params,
                      heatmap_size=64, image_size=256, sigma=2):
    """키포인트 좌표를 2D 가우시안 히트맵으로 변환합니다."""
    num_joints = len(joints)
    heatmaps = np.zeros((num_joints, heatmap_size, heatmap_size), dtype=np.float32)
    scale_factor = heatmap_size / image_size

    for idx in range(num_joints):
        if joints_vis[idx] == 0:
            continue

        crop_joint = transform_joint_to_crop(joints[idx], transform_params)
        hm_x = crop_joint[0] * scale_factor
        hm_y = crop_joint[1] * scale_factor

        if hm_x < 0 or hm_x >= heatmap_size or hm_y < 0 or hm_y >= heatmap_size:
            continue

        x = np.arange(0, heatmap_size, 1, np.float32)
        y = x[:, np.newaxis]
        g = np.exp(-((x - hm_x) ** 2 + (y - hm_y) ** 2) / (2 * sigma ** 2))
        heatmaps[idx] = g

    return heatmaps


def generate_synthetic_image(ann, output_dir):
    """Annotation 정보를 기반으로 합성 이미지를 생성합니다."""
    img_name = ann['image']
    img_path = os.path.join(output_dir, img_name)

    if os.path.exists(img_path):
        return img_path

    center = np.array(ann['center'])
    scale = ann['scale']
    body_size = scale * 200

    h = min(max(int(center[1] + body_size), 480), 1000)
    w = min(max(int(center[0] + body_size * 0.6), 640), 1000)

    img = np.random.randint(30, 80, (h, w, 3), dtype=np.uint8)

    joints = np.array(ann['joints'])
    joints_vis = ann['joints_vis']

    colors = [
        (255, 100, 100), (255, 150, 100), (255, 200, 100),
        (100, 255, 100), (100, 255, 150), (100, 255, 200),
        (200, 200, 200), (255, 255, 200), (200, 255, 255), (255, 200, 255),
        (100, 100, 255), (100, 150, 255), (100, 200, 255),
        (200, 100, 255), (200, 150, 255), (200, 200, 255),
    ]

    for (i, j) in SKELETON:
        if joints_vis[i] == 1 and joints_vis[j] == 1:
            pt1 = tuple(joints[i].astype(int))
            pt2 = tuple(joints[j].astype(int))
            if 0 <= pt1[0] < w and 0 <= pt1[1] < h and 0 <= pt2[0] < w and 0 <= pt2[1] < h:
                cv2.line(img, pt1, pt2, (180, 180, 180), 3)

    for idx, (joint, vis) in enumerate(zip(joints, joints_vis)):
        if vis == 1:
            x, y = int(joint[0]), int(joint[1])
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(img, (x, y), 8, colors[idx % len(colors)], -1)
                cv2.circle(img, (x, y), 10, (255, 255, 255), 2)

    img = cv2.GaussianBlur(img, (5, 5), 0)
    cv2.imwrite(img_path, img)
    return img_path


class MPIIDataset(Dataset):
    """MPII Human Pose Dataset (PyTorch Dataset)"""

    def __init__(self, annotations, image_dir, image_size=256, heatmap_size=64, sigma=2):
        self.annotations = annotations
        self.image_dir = image_dir
        self.image_size = image_size
        self.heatmap_size = heatmap_size
        self.sigma = sigma
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        parsed = parse_one_annotation(ann)

        img_path = os.path.join(self.image_dir, parsed['image'])
        img = cv2.imread(img_path)
        if img is None:
            img = np.zeros((480, 640, 3), dtype=np.uint8)

        cropped, params = crop_image(img, parsed['center'], parsed['scale'], self.image_size)
        cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)

        heatmaps = generate_heatmaps(
            parsed['joints'], parsed['joints_vis'], params,
            self.heatmap_size, self.image_size, self.sigma
        )

        image_tensor = self.transform(cropped_rgb).float()
        heatmap_tensor = torch.from_numpy(heatmaps).float()
        joints_vis_tensor = torch.from_numpy(parsed['joints_vis']).float()

        return image_tensor, heatmap_tensor, joints_vis_tensor


def heatmaps_to_keypoints(heatmaps, image_size=256):
    """히트맵에서 키포인트 좌표를 추출합니다."""
    num_joints, h, w = heatmaps.shape
    keypoints = np.zeros((num_joints, 2))
    confidences = np.zeros(num_joints)

    for j in range(num_joints):
        hm = heatmaps[j]
        conf = hm.max()
        confidences[j] = conf
        if conf > 0:
            idx = hm.argmax()
            y, x = divmod(idx, w)
            keypoints[j, 0] = x * image_size / w
            keypoints[j, 1] = y * image_size / h

    return keypoints, confidences


def draw_pose(image, keypoints, confidences=None, threshold=0.1):
    """이미지 위에 pose를 그립니다."""
    img = image.copy()

    colors = [
        (255, 85, 85), (255, 120, 85), (255, 170, 85),
        (85, 255, 85), (85, 255, 120), (85, 255, 170),
        (200, 200, 200), (255, 255, 170), (170, 255, 255), (255, 170, 255),
        (85, 85, 255), (85, 120, 255), (85, 170, 255),
        (170, 85, 255), (170, 120, 255), (170, 170, 255),
    ]

    for (i, j) in SKELETON:
        if confidences is not None:
            if confidences[i] < threshold or confidences[j] < threshold:
                continue
        pt1 = tuple(keypoints[i].astype(int))
        pt2 = tuple(keypoints[j].astype(int))
        cv2.line(img, pt1, pt2, (255, 255, 255), 2)

    for idx, kp in enumerate(keypoints):
        if confidences is not None and confidences[idx] < threshold:
            continue
        x, y = int(kp[0]), int(kp[1])
        cv2.circle(img, (x, y), 4, colors[idx % len(colors)], -1)
        cv2.circle(img, (x, y), 5, (255, 255, 255), 1)

    return img
