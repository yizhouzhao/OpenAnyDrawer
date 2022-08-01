import numpy as np
import cv2
import os

import torch
from torch.utils.data import DataLoader, Dataset

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

# Albumentations
def get_train_transform():
    return A.Compose([
        A.Resize(224, 224),
        A.Flip(0.5),
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

def get_valid_transform():
    return A.Compose([
        A.Resize(224, 224),
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})


def collate_fn(batch):
    return tuple(zip(*batch))
    
class HandleDataset(Dataset):

    def __init__(self, image_dir, transforms=None):
        super().__init__()

        self.image_dir = image_dir
        self.transforms = transforms

        self.get_img_ids()

    def get_img_ids(self):
        self.image_ids = []
        for image_id in os.listdir(self.image_dir):
            boxes_np = np.load(f'{self.image_dir}/{image_id}/bounding_box_2d_tight_0.npy')
            
            if boxes_np.shape[0] > 0:
                boxes = np.array([ list(e) for e in boxes_np])
                boxes = boxes[...,1:] # 0 is the class index

                boxes[:, 2:] += 1 # make max a little large

                self.image_ids.append([image_id, boxes])

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index: int):

        image_id, boxes = self.image_ids[index]

        image = cv2.imread(f'{self.image_dir}/{image_id}/rgb_0.png', cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        area = torch.as_tensor(area, dtype=torch.float32)

        # there is only one class
        labels = torch.ones((len(boxes),), dtype=torch.int)
        
        # suppose all instances are not crowd
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int)
        
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        # target['masks'] = None
        target['image_id'] = torch.tensor([index])
        target['area'] = area
        target['iscrowd'] = iscrowd

        if self.transforms:
            sample = {
                'image': image,
                'bboxes': target['boxes'],
                'labels': labels
            }
            sample = self.transforms(**sample)
            image = sample['image']
            
            target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)

        return image, target, image_id
