
import numpy as np
from glob import glob
from PIL import Image
from torch.utils.data import Dataset
from typing import Optional, Callable
import cv2
from .utils import expanded_join
import matplotlib.pyplot as plt
from torchvision import transforms


class CropSegmentationDataset(Dataset):
    ROOT_PATH: str = "/net/ens/am4ip/datasets/project-dataset"
    id2cls: dict = {0: "background",
                    1: "crop",
                    2: "weed",
                    3: "partial-crop",
                    4: "partial-weed"}
    
    cls2id: dict = {"background": 0,
                    "crop": 1,
                    "weed": 2,
                    "partial-crop": 3,
                    "partial-weed": 4}

    def __init__(self, set_type: str = "train", transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 merge_small_items: bool = True,
                 remove_small_items: bool = False):
        """Class to load datasets for the Project.

        Remark: `target_transform` is applied before merging items (this eases data augmentation).

        :param set_type: Define if you load training, validation or testing sets. Should be either "train", "val" or "test".
        :param transform: Callable to be applied on inputs.
        :param target_transform: Callable to be applied on labels.
        :param merge_small_items: Boolean to either merge classes of small or occluded objects.
        :param remove_small_items: Boolean to consider as background class small or occluded objects. If `merge_small_items` is set to `True`, then this parameter is ignored.
        """
        super(CropSegmentationDataset, self).__init__()
        self.transform = transform
        self.target_transform = target_transform
        self.merge_small_items = merge_small_items
        self.remove_small_items = remove_small_items

        if set_type not in ["train", "val", "test"]:
            raise ValueError("'set_type has an unknown value. "
                             f"Got '{set_type}' but expected something in ['train', 'val', 'test'].")

        self.set_type = set_type
        images = glob(expanded_join(self.ROOT_PATH, set_type, "images/*"))
        images.sort()
        self.images = np.array(images)
        labels = glob(expanded_join(self.ROOT_PATH, set_type, "labels/*"))
        labels.sort()
        self.labels = np.array(labels)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index: int):
        input_img = Image.open(self.images[index], "r")
        target = Image.open(self.labels[index], "r")

        before_img = np.array(input_img)
        before_target = np.array(target)
        if self.transform is not None:
            input_img = self.transform(input_img)
        
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.merge_small_items:
            target[target == self.cls2id["partial-crop"]] = self.cls2id["crop"]
            target[target == self.cls2id["partial-weed"]] = self.cls2id["weed"]
        elif self.remove_small_items:
            target[target == self.cls2id["partial-crop"]] = self.cls2id["background"]
            target[target == self.cls2id["partial-weed"]] = self.cls2id["background"]

       

        # opencv_input_img = cv2.cvtColor(before_img, cv2.COLOR_RGB2BGR)
        # input_img = np.array(cv2.bilateralFilter(opencv_input_img, 11, 0.4, 5))
        # input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        # # target = cv2.bilateralFilter(before_target.astype(np.float32), 11, 0.4, 5)
        
        
        # back_to_tensor = transforms.ToTensor()
        # input_img = back_to_tensor(input_img)
        # # target = back_to_tensor(target)

        # ax = plt.subplot(1, 2, 1)
        # ax.imshow(before_img)
        # ax.axis('off')
        # ax = plt.subplot(1, 2, 2)
        # ax.imshow(np.array(input_img.permute(1, 2, 0) ))
        # ax.axis('off')
        # ax = plt.subplot(2, 2, 3)
        # ax.imshow(np.array(before_target)*60, cmap="gray")
        # ax.axis('off')
        # ax = plt.subplot(2, 2, 4)
        # ax.imshow(np.array(np.array(target).squeeze(0))*60, cmap="gray")
        # ax.axis('off')
        # plt.subplots_adjust(wspace=0, hspace=0)
        # plt.savefig(f'EDA/noise_injection{index}_ex.png')
        # ToTensor()
        return input_img, target

    def get_class_number(self):
        if self.merge_small_items or self.remove_small_items:
            return 3
        else:
            return 5
