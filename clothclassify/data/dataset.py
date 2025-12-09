import functools
import glob
import random
import itertools

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
        self._list_file_with_label = self.build_labels(files)
        
        if train and val_stride > 0:
            del self._list_file_with_label[::val_stride]
            assert self._list_file_with_label, 'dataset empty'
        elif not train:
            assert val_stride > 0, f'val_stride cant = {val_stride}'
            self._list_file_with_label = \
                self._list_file_with_label[::val_stride]
    
    def build_labels(self, files: list[str]):
        """Getting image classes

        Args:
            files (list[str]): paths list

        Returns:
            labels (list[int]): labels for imgs
        """
        files_with_labels = []
        for file in files:
            cls_name = file.split('/')[-2]
            files_with_labels.append((file, self.classes[cls_name]))
        
        return files_with_labels
    
    # def shuffleDataset(self):
    #     """Shuffle dataset
    #     """
    #     random.shuffle(self._list_file_with_label)
    
    def __len__(self):
        if self.transform is not None and self.transform_count > 1:
            return len(self._list_file_with_label) * self.transform_count
        else:
            return len(self._list_file_with_label)
    
    def __getitem__(self, index):
        # if index == len(self._list_file_with_label) - 1:
        #     self.shuffleDataset()
        
        ndx = index % len(self._list_file_with_label)
        file, label = self._list_file_with_label[ndx]
        
        img = Image.open(file).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img, label


class TripletDataset(Dataset):
    def __init__(
        self,
        dir,
        train=True,
        val_stride=10,
        transform=None,
        transform_count=1
    ):
        """Image Dataset for Triplet Loss

        Args:
            dir (str): data directory
            train (bool, optional): train of val data. Defaults to True.
            val_stride (int, optional): val stride. Defaults to 10.
            transform (_type_, optional): transform for image. Defaults to None.
            transform_count (int, optional): count image. Defaults to 1.
        """
        super(TripletDataset, self).__init__()
        self.dir = dir
        self.transform = transform
        self.transform_count = transform_count
        self.classes, files = getFilesList(self.dir)
        
        self._dict_file_with_label = self.build_index(files)
        
        if train and val_stride > 0:
            self._dict_file_with_label = \
                {k: v for k, v in self._dict_file_with_label.items()
                 if k == 0 or k % val_stride != 0}
            assert self._dict_file_with_label, 'dataset empty'
        elif not train:
            assert val_stride > 0, f'val_stride cant = {val_stride}'
            self._dict_file_with_label = \
                {k: v for k, v in self._dict_file_with_label.items()
                 if k != 0 and k % val_stride == 0}
        
        self.length = sum(
            [len(v) for v in self._dict_file_with_label.values()]
        )
        self.files = list(
            itertools.chain.from_iterable(self._dict_file_with_label.values())
        )
    
    def build_index(self, files: list[str]):
        """Getting a dictionary of paths by index

        Args:
            files (list[str]): paths list

        Returns:
            files_index (dict[int, list]): _description_
        """
        files_index = {}
        for file in files:
            cls_label = self.path2id(file)
            if cls_label not in files_index:
                files_index[cls_label] = []
            files_index[cls_label].append(file)
        
        return files_index
    
    def path2id(self, path: str):
        """_summary_

        Args:
            path (str): path to file

        Returns:
            str: class name
        """
        return self.classes[path.split('/')[-2]]
    
    def __len__(self):
        if self.transform is None or self.transform_count == 1:
            return self.length
        else:
            return self.length * 3
    
    def __getitem__(self, index: int):
        ndx = index % len(self.files)
        anchor_path = self.files[ndx]
        
        positive_path = self.find_positive(anchor_path)
        negative_path = self.find_negative(anchor_path)
        
        anchor = Image.open(anchor_path)
        positive = Image.open(positive_path)
        negative = Image.open(negative_path)
        
        if self.transform is not None:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)
        
        return anchor, positive, negative
    
    def find_positive(self, anchor_path: str):
        """Getting positive img for path

        Args:
            anchor_path (str): anchor path

        Returns:
            positive_path (str): positive path
        """
        id = self.path2id(anchor_path)
        all_except_my = self._dict_file_with_label[id].copy()
        all_except_my.remove(anchor_path)
        positive_path = random.choice(all_except_my)
        return positive_path
    
    def find_negative(self, anchor_path):
        """Getting negative img for path

        Args:
            anchor_path (str): anchor path

        Returns:
            negative_path (str): negative path
        """
        id = self.path2id(anchor_path)
        all_except_my_ids = list(self._dict_file_with_label.keys())
        all_except_my_ids.remove(id)
        selected_id = random.choice(all_except_my_ids)
        negative_path = random.choice(self._dict_file_with_label[selected_id])
        return negative_path
