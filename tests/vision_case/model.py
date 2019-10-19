# coding=utf-8
# Created by Meteorix at 2019/8/9
import io
import os
import json
import torch
from torchvision import models
from torchvision import transforms
from PIL import Image

DIR_PATH = os.path.dirname(os.path.abspath(__file__))


class VisionModel(object):
    def __init__(self, device="cpu"):
        self.imagenet_class_index = json.load(open(os.path.join(DIR_PATH, 'imagenet_class_index.json')))
        self.device = device

    @staticmethod
    def transform_image(image_bytes):
        my_transforms = transforms.Compose([transforms.Resize(255),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize(
                                                [0.485, 0.456, 0.406],
                                                [0.229, 0.224, 0.225])])
        image = Image.open(io.BytesIO(image_bytes))
        return my_transforms(image).unsqueeze(0)

    def batch_prediction(self, image_bytes_batch):
        image_tensors = [self.transform_image(image_bytes=image_bytes) for image_bytes in image_bytes_batch]
        tensor = torch.cat(image_tensors).to(self.device)
        outputs = self.model.forward(tensor)
        _, y_hat = outputs.max(1)
        predicted_ids = y_hat.tolist()
        return [self.imagenet_class_index[str(i)] for i in predicted_ids]


class VisionDensenetModel(VisionModel):
    def __init__(self, device="cpu"):
        super().__init__(device=device)
        self.model = models.densenet121(pretrained=True)
        self.model.to(self.device)
        self.model.eval()


class VisionResNetModel(VisionModel):
    def __init__(self, device="cpu"):
        super().__init__(device=device)
        self.model = models.resnet101(pretrained=True)
        self.model.to(self.device)
        self.model.eval()
