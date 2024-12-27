from datasets.few_shot_dataset import FewShotDataset
import os
from typing import List
from PIL import Image
import numpy as np
from torchvision import transforms
Image.MAX_IMAGE_PIXELS = None
class Datasets(FewShotDataset):
     def __init__(self, args, transform, split='train'):
         data_dir = os.path.join(args.dataset_dir, split)
         images_path = []
         labels = []
         self.class_names = []
         if transform is None:
             self.transforms = transforms.Compose([
                 transforms.Resize(256),
                 transforms.RandomResizedCrop(240),
                 transforms.CenterCrop(224),
                 transforms.RandomHorizontalFlip(),
                 transforms.ToTensor(),
                transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])
         else:
             self.transform = transform

         if os.path.isdir(data_dir):
            for idx, label in enumerate(os.listdir(data_dir)):
                self.class_names.append(label)
                label_dir = os.path.join(data_dir, label)
                for image_path in os.listdir(label_dir):
                    images_path.append(os.path.join(label_dir, image_path))
                    labels.append(idx)
         else:
             raise ValueError('wrong dataset structure')
         self.class_to_label = {v: k for k, v in enumerate(self.class_names)}
         self.images_path = images_path
         self.labels = labels

     def __getitem__(self, item):
         path, label = self.images_path[item], self.labels[item]
         image = self.transforms(Image.open(path).convert('RGB'))
         return image, label

     def __len__(self):
         return len(self.images_path)

     def get_labels(self) -> List[int]:
         return self.labels




