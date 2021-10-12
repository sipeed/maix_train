

from torch.utils import data
from .net import SlimYOLOv2
from . import tools
import torch
import torch.optim as optim
from train_base import Train_Base
import os
from .eval import Evaluator

class Train(Train_Base):
    def __init__(self, classes, anchors, batch_size, input_size, input_layout, temp_dir, log, device="cuda"):
        super().__init__(classes, anchors, batch_size, input_size, input_layout, temp_dir, log, device=device)
        self.classes = classes
        self.log = log
        self.input_size = input_size
        if not anchors:
            log.e("anchors not valid: {}, should be [[w, h], ]".format(anchors))
            raise ValueError()
        self.anchors = anchors
        hr = True if input_size[0] > 400 else False
        self.device = device
        self.net = SlimYOLOv2(device, input_size=input_size, num_classes=len(classes), trainable=True, anchors=anchors, hr=hr)

    def train(self, dataset_root, dataset, dataloader, val_dataset, epoch, eval_every_epoch, save_every_epoch, lr_cb, loss_log_cb, early_stop, resume=None,
                momentum = 0.9, weight_decay = 5e-4, log_every_iter = 0,
                ingnore_thresh = 0.5):
        self.evaluator = Evaluator(dataset_root, self.device, val_dataset, self.classes)
        self.net.to(self.device).train()
        if resume:
            self.log.i("load pretrain parameters from {}".format(resume))
            self.net.load_state_dict(torch.load(resume, map_location=self.device))
        self.log.i("start train")
        lr = lr_cb(0, epoch)
        optimizer = optim.SGD(self.net.parameters(), 
                        lr=lr, 
                        momentum=momentum,
                        weight_decay= weight_decay
                        )
        if log_every_iter <= 0:
            log_every_iter = int(len(dataloader) / 10)
        max_iter = len(dataloader)
        for e in range(epoch):
            lr = lr_cb(e, epoch)
            self.set_lr(optimizer, lr)
            self.log.i("train epoch {}, lr: {}".format(e, lr))
            for iter_i, (images, targets) in enumerate(dataloader):
                images = images.to(self.device)
                targets = [label.tolist() for label in targets]
                targets = tools.gt_creator(input_size=self.input_size, 
                            stride=self.net.stride, 
                            label_lists=targets, 
                            anchor_size=self.anchors,
                            ingnore_thresh = ingnore_thresh
                            )
                targets = torch.tensor(targets).float().to(self.device)
                conf_loss, cls_loss, txtytwth_loss, total_loss = self.net(images, target=targets)
                total_loss.backward()        
                optimizer.step()
                optimizer.zero_grad()
                if iter_i % log_every_iter == 0 or (iter_i + 1) == len(dataloader):
                    loss_info = {
                        "conf": conf_loss.item(),
                        "class": cls_loss.item(),
                        "box": txtytwth_loss.item(),
                        "total": total_loss.item()
                        }
                    self.log.i("train epoch {}/{} iter {}/{}, loss(conf:{:.2f}, class:{:.2f}, box:{:.2f}, total:{:.2f}), remain {}".format(e, epoch, iter_i, len(dataloader),
                                loss_info["conf"], loss_info["class"], loss_info["box"], loss_info["total"],
                                self.estimated_remain_time(e, epoch, iter_i, len(dataloader)) )
                                )
                    loss_log_cb(e, epoch, iter_i, max_iter, loss_info)  
            if (e + 1) % eval_every_epoch == 0:
                self.net.trainable = False
                # self.net.set_grid(val_size)
                self.net.eval()

                # evaluate
                mean_ap, classes_aps = self.evaluator.evaluate(self.net)
                loss_info = {
                    "map": mean_ap,
                    "aps": classes_aps
                }
                if mean_ap:
                    loss_log_cb(e + 1, epoch, 0, max_iter, loss_info, is_val = True)

                # convert to training mode.
                self.net.trainable = True
                # self.net.set_grid(train_size)
                self.net.train()

            if (e + 1) % save_every_epoch == 0:
                save_path = os.path.join(self.save_path, f'epoch_{e + 1}.pth')
                self.log.i('saving state, epoch: {}, save to: {}'.format(e + 1, save_path))
                torch.save(self.net.state_dict(), save_path)


    def set_lr(self, optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
