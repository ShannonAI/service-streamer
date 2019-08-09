# coding=utf-8
# Created by Meteorix at 2019/8/9
import io
import json
import torch
from torchvision import models
from torchvision import transforms
from PIL import Image


imagenet_class_index = json.load(open('imagenet_class_index.json'))
device = "cuda"
# Make sure to pass `pretrained` as `True` to use the pretrained weights:
model = models.densenet121(pretrained=True)
model.to(device)
# Since we are using our model only for inference, switch to `eval` mode:
model.eval()


def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)


def batch_prediction(image_bytes_batch):
    image_tensors = [transform_image(image_bytes=image_bytes) for image_bytes in image_bytes_batch]
    tensor = torch.cat(image_tensors).to(device)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    predicted_ids = y_hat.tolist()
    return [imagenet_class_index[str(i)] for i in predicted_ids]


if __name__ == "__main__":
    with open(r"cat.jpg", 'rb') as f:
        image_bytes = f.read()

    print(batch_prediction([image_bytes] * 64))
