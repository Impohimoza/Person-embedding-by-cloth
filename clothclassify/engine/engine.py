import os
import shutil
import time
import datetime

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from clothclassify.data.datamanager import ImageDataManager
from clothclassify.utils import (
    AverageMeter, MetricMeter, open_all_layers, open_specified_layers, metrics
)


class Engine:
    def __init__(self, datamanager: ImageDataManager, use_gpu=True):
        """A generic base Engine class for both image-

        Args:
            datamanager (ImageDataManager): an instance of ``clothclassify.data.ImageDataManager``
            use_gpu (bool, optional): use gpu. Default is True.
        """
        self.datamanager = datamanager
        self.train_loader = datamanager.train_loader
        self.val_loader = datamanager.val_loader
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
        }
        dirpath = os.path.join(save_dir, 'checkpoint')
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        fpath = os.path.join(dirpath, 'model.pth.tar-' + str(epoch))
        torch.save(state, fpath)
        print('Checkpoint saved to "{}"'.format(fpath))
        if is_best:
            shutil.copy(
                fpath,
                os.path.join(os.path.dirname(fpath),
                             'model-best.pth.tar')
            )
    
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
        visrank_topk=10,
        ranks=[1, 5, 10, 20],
    ):
        """A unified pipeline for training and evaluating a model.

        Args:
            save_dir (str, optional): directory to save model.. Defaults to 'log'.
            max_epoch (int, optional): maximum epoch. Defaults to 0.
            start_epoch (int, optional): starting epoch. Defaults to 0.
            print_freq (int, optional): print_frequency. Defaults to 10.
            fixbase_epoch (int, optional): number of epochs to train ``open_layers`` (new layers)
                while keeping base layers frozen. Default is 0. ``fixbase_epoch`` is counted
                in ``max_epoch``.
            open_layers (_type_, optional): ayers (attribute names) open for training. Defaults to None.
            start_eval (int, optional): from which epoch to start evaluation. Defaults to 0.
            eval_freq (int, optional): evaluation frequency. Defaults to -1(meaning evaluation
                is only performed at the end of training).
            test_only (bool, optional): if True, only runs evaluation on test datasets. Defaults to False.
            dist_metric (str, optional): distance metric used to compute distance matrix. Defaults to 'euclidean'.
            normalize_feature (bool, optional): performs L2 normalization on feature vectors before
                computing feature distance. Defaults to False.
            visrank_topk (int, optional): top-k ranked images to be visualized. Default is 10.
            ranks (list, optional): cmc ranks to be computed. Default is [1, 5, 10, 20].
        """
        if test_only:
            self.test(
                dist_metric=dist_metric,
                normalize_feature=normalize_feature,
                save_dir=save_dir,
                ranks=ranks,
                visrank_topk=visrank_topk
            )
            return
        
        if self.writer is None:
            self.writer = SummaryWriter(
                log_dir=os.path.join(save_dir, 'tensorlog')
            )
        
        time_start = time.time()
        self.start_epoch = start_epoch
        self.max_epoch = max_epoch
        
        self.top_rank1 = 0
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
                    ranks=ranks,
                    visrank_topk=visrank_topk,
                )
                if self.top_rank1 <= rank1:
                    self.top_rank1 = rank1
                    self.save_model(self.epoch, rank1, save_dir, True)
                else:
                    self.save_model(self.epoch, rank1, save_dir)
                    
        if self.max_epoch > 0:
            print('=> Final test')
            rank1 = self.test(
                dist_metric=dist_metric,
                normalize_feature=normalize_feature,
                save_dir=save_dir,
                ranks=ranks,
                visrank_topk=visrank_topk
            )
            if self.top_rank1 <= rank1:
                self.top_rank1 = rank1
                self.save_model(self.epoch, rank1, save_dir, True)
            else:
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
                eta_seconds = batch_time.avg * \
                    (nb_this_epoch + nb_future_epochs)
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
                    self.writer.add_scalar(name, meter.avg, n_iter)
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
        ranks=[1, 5, 10, 20],
        visrank_topk=10
    ):
        self.set_model_mode('eval')
        print('##### Evaluating #####')
        
        rank1 = self._evaluate(
            self.val_loader,
            ranks=ranks,
            visrank_topk=visrank_topk,
        )
        
        return rank1
    
    @torch.no_grad()
    def _evaluate(
        self,
        dataloader,
        dist_metric='euclidean',
        normalize_feature=False,
        save_dir='',
        ranks=[1, 5, 10, 20],
        visrank_topk=10
    ):
        batch_time = AverageMeter()
        
        def _feature_extraction(data_loader):
            f_, pids_ = [], []
            for data in data_loader:
                imgs, pids = data
                if self.use_gpu:
                    imgs = imgs.cuda()
                end = time.time()
                features = self.model(imgs)
                batch_time.update(time.time() - end)
                features = features.cpu()
                f_.append(features)
                pids_.extend(pids.tolist())
            f_ = torch.cat(f_, 0)
            pids_ = np.asarray(pids_)
            return f_, pids_
        
        f, pids = _feature_extraction(dataloader)
        
        print('Speed: {:.4f} sec/batch'.format(batch_time.avg))
        
        if normalize_feature:
            f = F.normalize(f, p=2, dim=1)
        
        print(
            'Computing distance matrix with metric={} ...'.format(dist_metric)
        )
        
        distmat = metrics.compute_distance_matrix(f, f, dist_metric)
        distmat = distmat.numpy()
        
        print('Computing CMC...')
        
        cmc = metrics.evaluate_rank(distmat, pids, ranks)
        
        print('** Results **')
        
        print('CMC curve')
        for i, r in enumerate(ranks):
            print('Rank-{:<3}: {:.1%}'.format(r, cmc[r]))
        
        if self.writer is not None:
            self.writer.add_scalar('Test/rank1', cmc[1], self.epoch)
            self.writer.add_embedding(
                f,
                metadata=pids,
                global_step=self.epoch,
                tag='embedding'
            )
        
        return cmc[1]
    
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
            
        
class ImageSoftmaxEngine(Engine):
    def __init__(
        self,
        datamanager: ImageDataManager,
        model: nn.Module,
        optimizer,
        scheduler=None,
        use_gpu=True
    ):
        """Softmax-loss engine for image-reid.

        Args:
            datamanager (ImageDataManager): an instance of ``clothclassify.data.ImageDataManager``
            model (nn.Module): model instance.
            optimizer (Optimizer): an Optimizer.
            scheduler (LRScheduler, optional): if None, no learning rate decay will be performed.
            use_gpu (bool, optional): use gpu. Defaults to True.
        """
        super().__init__(datamanager, use_gpu)
        
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        self.criterion = nn.CrossEntropyLoss()
    
    def forward_backward(self, data):
        imgs, pids = data
        if self.use_gpu:
            imgs = imgs.cuda()
            pids = pids.cuda()
        
        outputs = self.model(imgs)
        loss = self.criterion(outputs, pids)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        loss_summary = {
            'loss': loss.item()
        }
        
        return loss_summary


class ImageTripletEngine(Engine):
    def __init__(
        self,
        datamanager: ImageDataManager,
        model: nn.Module,
        optimizer,
        scheduler=None,
        use_gpu=True
    ):
        """Triplet-loss engine for image-reid.

        Args:
            datamanager (ImageDataManager): an instance of ``clothclassify.data.ImageDataManager``
            model (nn.Module): model instance.
            optimizer (Optimizer): an Optimizer.
            scheduler (LRScheduler, optional): if None, no learning rate decay will be performed.
            use_gpu (bool, optional): use gpu. Defaults to True.
        """
        super().__init__(datamanager, use_gpu)
        
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        self.criterion = nn.TripletMarginLoss()
    
    def forward_backward(self, data):
        anchor, pos, neg = data
        if self.use_gpu:
            anchor = anchor.cuda()
            pos = pos.cuda()
            neg = neg.cuda()
        
        _, anchor_outputs = self.model(anchor)
        _, pos_outputs = self.model(pos)
        _, neg_outputs = self.model(neg)
        loss = self.criterion(anchor_outputs, pos_outputs, neg_outputs)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        loss_summary = {
            'loss': loss.item()
        }
        
        return loss_summary