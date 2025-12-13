import os
import shutil
import time
import datetime

import torch
from torch.utils.tensorboard import SummaryWriter

from clothclassify.data.datamanager import ImageDataManager
from clothclassify.utils import (
    AverageMeter, MetricMeter, open_all_layers, open_specified_layers
)


class Engine:
    def __init__(self, datamanager: ImageDataManager, use_gpu=True):
        self.datamanager = datamanager
        self.train_loader = self.train_loader
        self.val_loader = self.val_loader_loader
        self.use_gpu = use_gpu
        self.writer = None
        self.epoch = 0
        
        self.model_name = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
    
    def save_model(self, epoch, rank1, save_dir, is_best=False):
        state = {
            'state_dict': self.model.state_dict(),
            'epoch': epoch + 1,
            'rank1': rank1,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }
        fpath = os.path.join(save_dir, 'model.pth.tar-' + str(epoch))
        torch.save(state, fpath)
        print('Checkpoint saved to "{}"'.format(fpath))
        if is_best:
            shutil.copy(fpath, os.path.join(os.path.dirname(fpath), 'model-best.pth.tar'))
    
    def set_model_mode(self, mode='train'):
        assert mode in ['train', 'eval', 'test']
        
        if mode == 'train':
            self.model.train()
        else:
            self.model.eval()
    
    def get_current_lr(self):
        return self.optimizer.param_groups[-1]['lr']
    
    def update_lr(self):
        if self.scheduler is not None:
            self.scheduler.step()
    
    def run(
        self,
        save_dir='log',
        max_epoch=0,
        start_epoch=0,
        print_freq=10,
        fixbase_epoch=0,
        open_layers=None,
        start_eval=0,
        eval_freq=-1,
        test_only=False,
        dist_metric='euclidean',
        normalize_feature=False,
    ):
        if test_only:
            self.test(
                dist_metric=dist_metric,
                normalize_feature=normalize_feature,
                save_dir=save_dir,
            )
            return
        
        if self.writer is None:
            self.writer = SummaryWriter(log_dir=save_dir)
        
        time_start = time.time()
        self.start_epoch = start_epoch
        self.max_epoch = max_epoch
        print('=> Start training')
        
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.train(
                print_freq=print_freq,
                fixbase_epoch=fixbase_epoch,
                open_layers=open_layers
            )
            
            if (self.epoch + 1) >= start_eval \
               and eval_freq > 0 \
               and (self.epoch + 1) % eval_freq == 0 \
               and (self.epoch + 1) != self.max_epoch:
                rank1 = self.test(
                    dist_metric=dist_metric,
                    normalize_feature=normalize_feature,
                    save_dir=save_dir,
                )
                self.save_model(self.epoch, rank1, save_dir)
                    
        if self.max_epoch > 0:
            print('=> Final test')
            rank1 = self.test(
                dist_metric=dist_metric,
                normalize_feature=normalize_feature,
                save_dir=save_dir,
            )
            self.save_model(self.epoch, rank1, save_dir)
        
        elapsed = round(time.time() - time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print('Elapsed {}'.format(elapsed))
        if self.writer is not None:
            self.writer.close()
    
    def train(self, print_freq=10, fixbase_epoch=0, open_layers=None):
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        
        self.set_model_mode('train')
        
        self.two_stepped_transfer_learning(
            self.epoch, fixbase_epoch, open_layers
        )
        
        self.num_batches = len(self.train_loader)
        end = time.time()
        
        for self.batch_idx, data in enumerate(self.train_loader):
            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(data)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)
            
            if (self.batch_idx + 1) % print_freq == 0:
                nb_this_epoch = self.num_batches - (self.batch_idx + 1)
                nb_future_epochs = (
                    self.max_epoch - (self.epoch + 1)
                ) * self.num_batches
                eta_seconds = batch_time.avg * (nb_this_epoch+nb_future_epochs)
                eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
                
                print(
                    'epoch: [{0}/{1}][{2}/{3}]\t'
                    'time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'eta {eta}\t'
                    '{losses}\t'
                    'lr {lr:.6f}'.format(
                        self.epoch + 1,
                        self.max_epoch,
                        self.batch_idx + 1,
                        self.num_batches,
                        batch_time=batch_time,
                        data_time=data_time,
                        eta=eta_str,
                        losses=losses,
                        lr=self.get_current_lr()
                    )
                )
            
            if self.writer is not None:
                n_iter = self.epoch * self.num_batches + self.batch_idx
                self.writer.add_scalar('Train/time', batch_time.avg, n_iter)
                self.writer.add_scalar('Train/data', data_time.avg, n_iter)
                for name, meter in losses.meters.items():
                    self.writer.add_scalar('Train/' + name, meter.avg, n_iter)
                self.writer.add_scalar(
                    'Train/lr', self.get_current_lr(), n_iter
                )
            
            end = time.time()
        self.update_lr()
    
    def forward_backward(self, data):
        raise NotImplementedError
    
    def test(
        self,
        dist_metric='euclidean',
        normalize_feature=False,
        save_dir='',
    ):
        pass

    def two_stepped_transfer_learning(
        self, epoch, fixbase_epoch, open_layers
    ):
        if self.model is None:
            return
        
        if (epoch + 1) <= fixbase_epoch and open_layers is not None:
            print(
                '* Only train {} (epoch: {}/{})'.format(
                    open_layers, epoch + 1, fixbase_epoch
                )
            )
            open_specified_layers(self.model, open_layers)
        
        else:
            open_all_layers(self.model)
            
        
