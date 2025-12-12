import torch

from clothclassify.data.transforms import build_transforms
from clothclassify.data.dataset import SoftmaxDataset, TripletDataset


class ImageDataManager:
    def __init__(
        self,
        root='',
        height=256,
        width=128,
        transforms=['random_flip'],
        norm_mean=None,
        norm_std=None,
        use_gpu=True,
        loss_fn='softmax',
        batch_size_train=32,
        batch_size_val=32,
        workers=4
    ):
        """Image data manager.

        Args:
            root (str, optional): root path to datasets.. Defaults to ''.
            height (int, optional): target image height. Defaults to 256.
            width (int, optional): target image width. Defaults to 128.
            transforms (list, optional): transformations applied to model training. Defaults to ['random_flip'].
            norm_mean (_type_, optional): data mean. Default is None (use imagenet mean).
            norm_std (_type_, optional): data std. Default is None (use imagenet std).
            use_gpu (bool, optional): use gpu. Defaults to True.
            loss_fn (str, optional): why dataset use "softmax" or "triplet". Defaults to 'softmax'.
            batch_size_train (int, optional): number of images in a training batch. Defaults to 32.
            batch_size_val (int, optional): number of images in a test batch. Defaults to 32.
            workers (int, optional): number of workers. Defaults to 4.

        Examples::

            datamanager = clothclassify.data.datamanager.ImageDataManager(
                root='path/to/reid-data',
                height=256,
                width=128,
                batch_size_train=32,
                batch_size_test=100
            )

            # return train loader of source data
            train_loader = datamanager.train_loader

            # return val loader of target data
            val_loader = datamanager.val_loader
        """
        if loss_fn not in ['softmax', 'triplet']:
            raise Exception("loss_fn most be 'softmax' or 'triplet'")
        self.height = height
        self.width = width
        
        self.transform_tr, self.transform_te = build_transforms(
            self.height,
            self.width,
            transforms=transforms,
            norm_mean=norm_mean,
            norm_std=norm_std
        )
        
        self.use_gpu = (torch.cuda.is_available() and use_gpu)
        
        if loss_fn == "softmax":
            train_set = SoftmaxDataset(
                root,
                transform=self.transform_tr,
                transform_count=len(transforms)
            )
            
            val_set = SoftmaxDataset(
                root,
                train=False,
                transform=self.transform_te
            )
        else:
            train_set = TripletDataset(
                root,
                transform=self.transform_tr,
                transform_count=len(transforms)
            )
            val_set = TripletDataset(
                root,
                train=False,
                transform=self.transform_te
            )
        
        self.train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=batch_size_train,
            shuffle=True,
            num_workers=workers,
            pin_memory=self.use_gpu
        )
        
        self.val_loader = torch.utils.data.DataLoader(
            val_set,
            batch_size=batch_size_val,
            num_workers=workers,
            pin_memory=self.use_gpu
        )
    
    def preprocess_img(self, img):
        return self.transform_te(img)
        
        