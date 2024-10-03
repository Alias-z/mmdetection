# Copyright (c) OpenMMLab. All rights reserved.
import os
import random
from typing import List, Union
from collections import defaultdict

from mmengine.fileio import get_local_path

from mmdet.registry import DATASETS
from .api_wrappers import COCO
from .base_det_dataset import BaseDetDataset


@DATASETS.register_module()
class CocoEevenSamplerDataset(BaseDetDataset):
    """Mutiple Datasets COCO that will sampled evenly"""

    METAINFO = {
        'classes':
        ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
         'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
         'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
         'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
         'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
         'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
         'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
         'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
         'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
         'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
         'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
         'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
         'scissors', 'teddy bear', 'hair drier', 'toothbrush'),
        # palette is a list of color tuples, which is used for visualization.
        'palette':
        [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
         (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
         (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30),
         (165, 42, 42), (255, 77, 255), (0, 226, 252), (182, 182, 255),
         (0, 82, 0), (120, 166, 157), (110, 76, 0), (174, 57, 255),
         (199, 100, 0), (72, 0, 118), (255, 179, 240), (0, 125, 92),
         (209, 0, 151), (188, 208, 182), (0, 220, 176), (255, 99, 164),
         (92, 0, 73), (133, 129, 255), (78, 180, 255), (0, 228, 0),
         (174, 255, 243), (45, 89, 255), (134, 134, 103), (145, 148, 174),
         (255, 208, 186), (197, 226, 255), (171, 134, 1), (109, 63, 54),
         (207, 138, 255), (151, 0, 95), (9, 80, 61), (84, 105, 51),
         (74, 65, 105), (166, 196, 102), (208, 195, 210), (255, 109, 65),
         (0, 143, 149), (179, 0, 194), (209, 99, 106), (5, 121, 0),
         (227, 255, 205), (147, 186, 208), (153, 69, 1), (3, 95, 161),
         (163, 255, 0), (119, 0, 170), (0, 182, 199), (0, 165, 120),
         (183, 130, 88), (95, 32, 0), (130, 114, 135), (110, 129, 133),
         (166, 74, 118), (219, 142, 185), (79, 210, 114), (178, 90, 62),
         (65, 70, 15), (127, 167, 115), (59, 105, 106), (142, 108, 45),
         (196, 172, 0), (95, 54, 80), (128, 76, 255), (201, 57, 1),
         (246, 0, 122), (191, 162, 208)]
    }
    COCOAPI = COCO
    # ann_id is unique in coco dataset.
    ANN_ID_UNIQUE = True

    def __init__(self,
                 all_datasets: list = None,
                 total_samples: int = None,
                 **kwargs):
        self.all_datasets = all_datasets
        self.total_samples = total_samples
        super().__init__(**kwargs)

    def load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file named as ``self.ann_file``

        Returns:
            List[dict]: A list of annotation.
        """  # noqa: E501
        with get_local_path(self.ann_file, backend_args=self.backend_args) as local_path:
            self.coco = self.COCOAPI(local_path)  # load the json file

        self.cat_ids = self.coco.get_cat_ids(cat_names=self.metainfo['classes'])  # get the catergories
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}  # map catergories to their ids
        self.cat_img_map = self.coco.cat_img_map

        is_train = 'train' in os.path.basename(self.ann_file).lower()   # determine whether we're loading train or val data
        subfolder = 'train_sahi' if is_train else 'val_sahi'  # to pick the train or val folder

        all_img_infos = self.coco.loadImgs(self.coco.getImgIds())  # get all image info
        basename_to_id = {os.path.basename(img['file_name']): img['id'] for img in all_img_infos}  # create a mapping of basenames to image IDs

        image_types = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.gif', '.ico', '.jfif', '.webp')  # define supported image types

        # Group images by dataset

        total_n_images = 0

        images_by_dataset = defaultdict(list)
        for dataset in self.all_datasets:
            dataset_path = os.path.join(self.data_root, dataset, subfolder)
            if os.path.exists(dataset_path):
                for root, _, files in os.walk(dataset_path):
                    for file in files:
                        if file.lower().endswith(image_types):
                            total_n_images += 1
                            images_by_dataset[dataset].append(file)

        # print(f'Total images from all datasets = {total_n_images}')

        samples_per_dataset = self.total_samples // len(self.all_datasets)  # calculate number of samples per dataset

        # Sample images from each dataset
        sampled_images = defaultdict(list)
        for dataset, images in images_by_dataset.items():
            if len(images) < samples_per_dataset:
                sampled_images[dataset] = images
            else:
                sampled_images[dataset] = random.sample(images, samples_per_dataset)

        # print(f"\nSampling Results for {'train' if is_train else 'val'} data:")
        # for dataset, images in sampled_images.items():
        #     print(f"\nDataset: {dataset}")
        #     print(f"Number of images sampled: {len(images)}")
        #     print("Sampled image names [the frist 5]:")
        #     for image in images[:5]:  # print first 5 image names
        #         print(f"  - {image}")
        #     if len(images) > 5:
        #         print(f"  ... and {len(images) - 5} more")

        all_sampled_images = [image for images in sampled_images.values() for image in images]  # flatten the sampled images list
        sampled_img_ids = [basename_to_id[basename] for basename in all_sampled_images if basename in basename_to_id]  # get the image IDs for the sampled images

        # Load the data for the sampled images
        data_list = []
        total_ann_ids = []
        for img_id in sampled_img_ids:
            raw_img_info = self.coco.loadImgs([img_id])[0]
            raw_img_info['img_id'] = img_id

            ann_ids = self.coco.getAnnIds(imgIds=[img_id])
            raw_ann_info = self.coco.loadAnns(ann_ids)
            total_ann_ids.extend(ann_ids)

            parsed_data_info = self.parse_data_info({
                'raw_ann_info': raw_ann_info,
                'raw_img_info': raw_img_info
            })
            data_list.append(parsed_data_info)

        if self.ANN_ID_UNIQUE:
            assert len(set(total_ann_ids)) == len(total_ann_ids), f"Annotation ids in '{self.ann_file}' are not unique!"

        # print(f"\nTotal images sampled: {len(data_list)}")
        return data_list

    def parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]:
        """Parse raw annotation to target format.

        Args:
            raw_data_info (dict): Raw data information load from ``ann_file``

        Returns:
            Union[dict, List[dict]]: Parsed annotation.
        """
        img_info = raw_data_info['raw_img_info']
        ann_info = raw_data_info['raw_ann_info']

        data_info = {}

        # Construct the full path to the image file
        img_path = os.path.join(self.data_root, img_info['file_name'])

        if self.data_prefix.get('seg', None):
            seg_map_path = os.path.join(
                self.data_prefix['seg'],
                img_info['file_name'].rsplit('.', 1)[0] + self.seg_map_suffix)
        else:
            seg_map_path = None
        data_info['img_path'] = img_path
        data_info['img_id'] = img_info['img_id']
        data_info['seg_map_path'] = seg_map_path
        data_info['height'] = img_info['height']
        data_info['width'] = img_info['width']

        if self.return_classes:
            data_info['text'] = self.metainfo['classes']
            data_info['caption_prompt'] = self.caption_prompt
            data_info['custom_entities'] = True

        instances = []
        for i, ann in enumerate(ann_info):
            instance = {}

            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]

            if ann.get('iscrowd', False):
                instance['ignore_flag'] = 1
            else:
                instance['ignore_flag'] = 0
            instance['bbox'] = bbox
            instance['bbox_label'] = self.cat2label[ann['category_id']]

            if ann.get('segmentation', None):
                instance['mask'] = ann['segmentation']

            instances.append(instance)
        data_info['instances'] = instances
        return data_info

    def filter_data(self) -> List[dict]:
        """Filter annotations according to filter_cfg.

        Returns:
            List[dict]: Filtered results.
        """
        if self.test_mode:
            return self.data_list

        if self.filter_cfg is None:
            return self.data_list

        filter_empty_gt = self.filter_cfg.get('filter_empty_gt', False)
        min_size = self.filter_cfg.get('min_size', 0)

        # obtain images that contain annotation
        ids_with_ann = set(data_info['img_id'] for data_info in self.data_list)
        # obtain images that contain annotations of the required categories
        ids_in_cat = set()
        for i, class_id in enumerate(self.cat_ids):
            ids_in_cat |= set(self.cat_img_map[class_id])
        # merge the image id sets of the two conditions and use the merged set
        # to filter out images if self.filter_empty_gt=True
        ids_in_cat &= ids_with_ann

        valid_data_infos = []
        for i, data_info in enumerate(self.data_list):
            img_id = data_info['img_id']
            width = data_info['width']
            height = data_info['height']
            if filter_empty_gt and img_id not in ids_in_cat:
                continue
            if min(width, height) >= min_size:
                valid_data_infos.append(data_info)

        return valid_data_infos
