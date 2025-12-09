import functools
import glob
import random

from torch.utils.data import Dataset
from PIL import Image


@functools.lru_cache(1)
def getFilesList(dir: str):
    """function for get classes and files list

    Args:
        dir (str): data directory

    Returns:
        folder_to_index: classes list
        file_list: path to file list
    """
    file_list = glob.glob(f"{dir}/*/*.jpg")
    
    folder_to_index = {folder.split('/')[-1]: i
                       for i, folder in enumerate(glob.glob(f"{dir}/*"))}
    return folder_to_index, file_list


class SoftmaxDataset(Dataset):
    def __init__(
        self,
        dir: str,
        train=True,
        val_stride=10,
        transform=None,
        transform_count=1
    ):
        """Class-Based Image Dataset

        Args:
            dir (str): data directory
            train (bool, optional): train of val data. Defaults to True.
            val_stride (int, optional): val stride. Defaults to 10.
            transform (_type_, optional): transform for image. Defaults to None.
            transform_count (int, optional): count image. Defaults to 1.
        """
        super(SoftmaxDataset, self).__init__()
        self.dir = dir
        self.transform = transform
        self.transform_count = transform_count
        self.classes, files = getFilesList(self.dir)
        labels = self.build_labels(files)
        
        self._list_file_with_label = list(zip(files, labels))
        
        if train and val_stride > 0:
            del self._list_file_with_label[::val_stride]
            assert self._list_file_with_label, 'dataset empty'
        elif not train:
            assert val_stride > 0, f'val_stride cant = {val_stride}'
            self._list_file_with_label = \
                self._list_file_with_label[::val_stride]
    
    def build_labels(self, files: list[str]):
        """getting image classes

        Args:
            files (list[str]): paths list

        Returns:
            labels (list[int]): labels for imgs
        """
        labels = []
        for file in files:
            cls_name = file.split('/')[-2]
            labels.append(self.classes[cls_name])
        
        return labels
    
    def shuffleDataset(self):
        """Shuffle dataset
        """
        random.shuffle(self._list_file_with_label)
    
    def __len__(self):
        if self.transform is not None and self.transform_count > 1:
            return len(self._list_file_with_label) * self.transform_count
        else:
            return len(self._list_file_with_label)
    
    def __getitem__(self, index):
        if index == len(self._list_file_with_label) - 1:
            self.shuffleDataset()
        
        ndx = index % len(self._list_file_with_label)
        file, label = self._list_file_with_label[ndx]
        
        img = Image.open(file).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img, label
        