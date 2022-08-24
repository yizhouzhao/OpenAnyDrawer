import torch

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN

from exp.learning.custom_cliport import CustomCliport

def load_vision_model(
    model_path = "/home/yizhou/Research/temp0/fasterrcnn_resnet50_fpn.pth", 
    model_name = "fasterrcnn_resnet50_fpn",
    clip_text_feature_path = "/home/yizhou/Research/OpenAnyDrawer/learning/text2clip_feature.json"
    ):
    # load a model; pre-trained on COCO
    if model_name == "fasterrcnn_resnet50_fpn":
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

        num_classes = 2  # 1 class (wheat) + background

        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features

        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        model.load_state_dict(torch.load(model_path))
        model.eval()

    elif model_name == "custom_cliport":
        model = CustomCliport(clip_text_feature_path = clip_text_feature_path)
        model.load_state_dict(torch.load(model_path))

        model = model.to(model.device)
        model.set_prediction_mode()
        model.eval()

    return model
