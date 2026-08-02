"""GradCAM visualization for the trained steering model.

Highlights which image regions most influence the steering prediction from the
EndtoEnd model trained by train.py -- useful for the thesis's Phase 2 "Feature
Visualization" deliverable (attention/saliency maps).

Requires the optional 'pytorch_grad_cam' package, not part of requirements.txt's
core pipeline dependencies:
    pip install grad-cam

Usage:
    Copy a few sample frames (same run<N>_images/*.png filename format) into an
    'example_imgs/' folder, then:
    python vis_gradcam.py --model_path saved_models_iter0
"""
import argparse
import glob
import os

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image, ImageFile
import matplotlib.pyplot as plt

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

ImageFile.LOAD_TRUNCATED_IMAGES = True

device = 'cpu'
IMAGE_SIZE = 256
WIDTH = 800
HEIGHT = 600
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

# Must match train.py's transform exactly, since GradCAM needs the same preprocessing
# the model was trained on.
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.CenterCrop(min(HEIGHT, WIDTH)),
    transforms.Resize(IMAGE_SIZE),
    transforms.Normalize(tuple(IMAGENET_MEAN), tuple(IMAGENET_STD))])


class EndtoEnd(models.resnet.ResNet):
    """Must stay in sync with the EndtoEnd class in train.py / run_iter.py."""
    def __init__(self):
        super(EndtoEnd, self).__init__(models.resnet.BasicBlock, [2, 2, 2, 2])
        self.speed_feat_extractor = nn.Linear(2, 64)
        self.final_layer = nn.Linear(64 + 128, 1)

    def forward(self, x, v, vperp):
        speed_feat = torch.cat([torch.atan2(vperp, v), torch.sqrt(v**2 + vperp**2)], dim=1)
        x1 = super(EndtoEnd, self).forward(x)
        x2 = self.speed_feat_extractor(speed_feat)
        x = torch.cat((x1, x2), axis=1)
        x = self.final_layer(x)
        return x


class ImageOnlySteeringModel(nn.Module):
    """Freezes (v, vperp) so GradCAM can treat the model as a plain image -> scalar function."""
    def __init__(self, model, v, vperp):
        super().__init__()
        self.model = model
        self.v = v
        self.vperp = vperp

    def forward(self, x):
        return self.model(x, self.v, self.vperp)


class ImageDataset(Dataset):
    """Same filename-encoded-label convention as train.py's croppedDataset."""
    def __init__(self, ims):
        self.ims = ims

    def __len__(self):
        return len(self.ims)

    def __getitem__(self, index):
        image_path = self.ims[index]
        image = Image.open(image_path)
        X = transform(image)
        base_name = os.path.basename(image_path)
        Y = base_name.split('.')[0].split('_')
        v = float(Y[6]) / 100.
        vperp = float(Y[7]) / 100.
        return X.float(), torch.tensor([v]).float(), torch.tensor([vperp]).float()


def denormalize_for_display(image_chw_tensor):
    img = image_chw_tensor.permute(1, 2, 0).numpy()
    img = img * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(img, 0, 1)


def main(args):
    image_paths = sorted(glob.glob(args.image_folder + '/*.png'))
    if not image_paths:
        raise SystemExit(f"No images found in {args.image_folder}/ -- copy a few sample frames there first.")
    loader = DataLoader(ImageDataset(image_paths), batch_size=1, shuffle=False)

    model = EndtoEnd()
    model.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(model.fc.in_features, 128))
    model.load_state_dict(torch.load(os.path.join(args.model_path, 'model-last.ckpt'), map_location=device))
    model.eval()

    target_layers = [model.layer4[-1]]
    os.makedirs('results/gradcam', exist_ok=True)

    for i, (image, v, vperp) in enumerate(loader):
        wrapped = ImageOnlySteeringModel(model, v, vperp)
        cam = GradCAM(model=wrapped, target_layers=target_layers)
        grayscale_cam = cam(input_tensor=image)[0]
        display_image = denormalize_for_display(image[0])
        visualization = show_cam_on_image(display_image, grayscale_cam, use_rgb=True)
        out_path = f'results/gradcam/frame_{i}.png'
        plt.imsave(out_path, visualization)
        print(f"Saved {out_path}")


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('--model_path', default='saved_models_iter0',
                            help='Directory containing model-last.ckpt (from train.py)')
    argparser.add_argument('--image_folder', default='example_imgs',
                            help='Folder of sample frames to visualize')
    args = argparser.parse_args()
    main(args)
