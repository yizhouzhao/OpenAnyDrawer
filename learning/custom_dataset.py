import numpy as np
import cv2
import os
from PIL import Image, ImageDraw

import json

import torch
from torch.utils.data import DataLoader, Dataset

from tqdm.auto import tqdm

# load vision model
from transformers import AutoFeatureExtractor, ResNetModel
feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/resnet-18")
resnet_model = ResNetModel.from_pretrained("microsoft/resnet-18").to("cuda")

# load language feature
import pickle
text2clip_feature = pickle.load(open("text2clip_feature.pickle",'rb'))

def collate_fn(batch):
    image_list = []
    target_list = []
    text_feature_list = []

    for (image, target, text) in batch:
        image_list.append(image)
        target_list.append(torch.tensor(target))
        text_feature_list.append(text2clip_feature[text])
    
    # image features
    inputs = feature_extractor(image_list, return_tensors="pt").to("cuda")
    with torch.no_grad():
        image_features = resnet_model(**inputs).last_hidden_state

    # targets
    targets = torch.stack(target_list).to("cuda")
    text_feautures = torch.stack(text_feature_list).to("cuda")

    return image_features.float(), targets.float(), text_feautures.float()
    


class HandleDataset4Cliport(Dataset):

    def __init__(self, image_dir, num_frames = 5, is_train = True, transforms=None):
        super().__init__()

        self.image_dir = image_dir
        self.num_frames = num_frames # randomized frames in rendering
        self.transforms = transforms
        self.is_train = is_train

        self.get_img_ids()

    def get_img_ids(self):
        self.image_ids = []
        for image_id in tqdm(sorted(os.listdir(self.image_dir), key = lambda x: int(x))):
    
            if self.is_train:
                if int(image_id) > 150:
                    continue
            else: # test
                if int(image_id) <= 150:
                    continue

            # print("image_id", image_id)
            for i in range(self.num_frames):
                boxes_np = np.load(f'{self.image_dir}/{image_id}/bounding_box_2d_tight_{i}.npy')
                lang_json = json.load(open(f'{self.image_dir}/{image_id}/bounding_box_2d_tight_labels_{i}.json'))

                if boxes_np.shape[0] > 0:
                    boxes = np.array([ list(e) for e in boxes_np])
                    boxes = boxes[...,1:] # 0 is the class index
                    boxes[:, :2] -= 1 # make min a little smaller
                    boxes[:, 2:] += 1 # make max a little large
                
                for j, key in enumerate(lang_json):
                    self.image_ids.append([image_id, boxes[j], i, lang_json[key]['class']]) # image, box, frame, text

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index: int):
        """
        return:
            image: image
        """

        image_id, box, frame, text = self.image_ids[index]
        image = Image.open(f'{self.image_dir}/{image_id}/rgb_{frame}.png')
        image = image.convert('RGB')

        box_image = Image.new('L', image.size)
        draw_image = ImageDraw.Draw(box_image)  
        draw_image.rectangle(list(box), fill ="#FFFFFF")
        box_image = np.array(box_image) / 255.0

        text = text.replace("_"," ").replace("-"," ").replace("  ", " ").strip()

        return image, box_image, text


