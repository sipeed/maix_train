import os, sys
import cv2
import random
import time

from torch.utils.tensorboard import SummaryWriter

try:
    from .dataset import Dataset_VOC
    from .dataloader import DataLoader
    from .logger import Logger
    from .augmentations import SSDAugmentation, DeNormalize, SSDAugmentationTest
    from .draw import Draw
except Exception:
    from dataset import Dataset_VOC
    from dataloader import DataLoader
    from logger import Logger
    from augmentations import SSDAugmentation, DeNormalize, SSDAugmentationTest
    from draw import Draw

class Train:
    def __init__(self, classes, net_type, dataset_name, batch_size, input_shape=(3, 416, 416), anchors = None, input_layout="default", temp_dir=None, opt = {}, log = Logger(), device="cuda"):
        '''
            @input_layout only support default now, pytorch is chw, tensorflow is hwc
        '''
        self.classes = classes
        self.net_type = net_type
        self.log = log
        self.input_shape = input_shape
        if not temp_dir:
            temp_dir = os.path.join("out", dataset_name, net_type)
        self.temp_dir = temp_dir
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)
        self.batch_size = batch_size
        self.root = os.path.abspath(os.path.dirname(__file__))
        detectors_path = os.path.join(self.root, "detectors")
        sys.path.insert(0, detectors_path)
        self.detector = __import__(net_type)
        try:
            self.framework = self.detector.framwork
        except Exception:
            self.framework = "torch"
        if input_layout != "default":
            raise NotImplementedError()
        self.input_layout = ("chw" if self.framework == "torch" else "hwc") if input_layout == "default" else input_layout
        self.anchors = self.val_anchors(anchors)
        self.trainer = self.detector.Train(
                                         classes = classes,
                                         anchors = self.anchors,
                                         batch_size = batch_size,
                                         input_size=(self.input_shape[2],
                                         self.input_shape[1]),
                                         input_layout = self.input_layout,
                                         temp_dir = self.temp_dir,
                                         log = self.log,
                                         device=device)
        self.draw = Draw(self.classes)
        c_time = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
        log_path = os.path.join(self.temp_dir, "logs", c_time)
        os.makedirs(log_path, exist_ok=True)
        self.log_writer = SummaryWriter(log_path)
        self.last_epoch = -1

    def load_dataset(self, path, format = "voc", load_num_workers = 4, dataset = None):
        '''
            load dataset as iter object, item:
                [(img, target), ...]
            and as loader:
                [batch_imgs, batch_targets]
        '''
        if format != "voc":
            raise NotImplementedError()
        self.dataset_path = path
        self.dataset = dataset
        if not dataset:
            self.dataset = Dataset_VOC(self.classes, path, sets=["train"], log = self.log,
                               transform = SSDAugmentation(size=(self.input_shape[2], self.input_shape[1]), mean=(0.5, 0.5, 0.5), std=(128/255.0, 128/255.0, 128/255.0))
                               )
        self.dataloader = DataLoader(self.framework).get_dataloader(self.dataset, self.batch_size, num_workers=load_num_workers)
        self.val_dataset = Dataset_VOC(self.classes, path, sets=["val"], log = self.log,
                               transform = SSDAugmentationTest(size=(self.input_shape[2], self.input_shape[1]), mean=(0.5, 0.5, 0.5), std=(128/255.0, 128/255.0, 128/255.0))
                               )
        self.log.i("dataset length: ", len(self.dataset))


    def train(self, epoch, eval_every_epoch, save_every_epoch, lr_cb=None, early_stop = False, resume = None):
        self.epoch = epoch
        self.eval_every_epoch = eval_every_epoch
        if not lr_cb:
            lr_cb = self.default_lr
        self.trainer.train(self.dataset_path, self.dataset, self.dataloader, self.val_dataset, epoch, eval_every_epoch, save_every_epoch,
                        lr_cb = lr_cb, loss_log_cb = self.on_loss_log,
                        early_stop = early_stop,
                        resume = resume)
        self.log.i("train complete")

    def eval(self, size=(224, 224)):
        pass

    def test(self):
        test_dataset = Dataset_VOC(self.classes, self.dataset_path, sets=["test"], log = self.log,
                        transform = SSDAugmentation(size=(self.input_shape[2], self.input_shape[1]), mean=(0.5, 0.5, 0.5), std=(128/255.0, 128/255.0, 128/255.0))
                        )

    def default_lr(self, epoch, max_epoch):
        if epoch == 0:
            return 1e-3
        if epoch < int(max_epoch * 0.3):
            return 0.0005
        if epoch < int(max_epoch * 0.5):
            return 1e-4
        return 1e-5

    def on_loss_log(self, epoch, max_epoch, iter_i, max_iter, record, is_val = False):
        '''
            @record dict:
                    {
                        "conf": conf_loss.item(),
                        "class": cls_loss.item(),
                        "box": txtytwth_loss.item(),
                        "total": total_loss.item()
                    }
                    {
                        "map": float,
                        "aps": [(class_name1, float), ...]
                    }
        '''
        # self.log.i(f'train epoch {epoch}: {record}')
        if is_val:
            scalars = {
                "mAP": record["map"]
            }
            for cls, ap in record["aps"]:
                scalars[cls] = ap
            print(scalars)
            self.log_writer.add_scalars('AP', scalars, iter_i + epoch * max_iter)
        else:
            self.log_writer.add_scalars('loss', record, iter_i + epoch * max_iter)
        if self.last_epoch != epoch:
            self.log_writer.flush()
            self.last_epoch = epoch


    def report(self):
        pass

    def preview_data(self):
        for i in range(len(self.dataset)):
            img, target = self.dataset.pull_item2(i, denorm=DeNormalize(mean=(0.5, 0.5, 0.5), std=(128/255.0, 128/255.0, 128/255.0)))
            labels = []
            for gt in target:
                labels.append(gt[-1])
            img = self.draw.draw_img(img, target, labels, self.classes)
            path = os.path.join(self.temp_dir, "img.jpg")
            cv2.imwrite(path, img)
            input(f"see {path}, press any key to continue")

    def val_anchors(self, anchors):
        '''
            convert [w, h, w, h, ...] to [[w, h], [w, h], ...]
        '''
        if type(anchors[0]) == list or type(anchors[0]) == tuple:
            return anchors
        final = []
        for i in range(0, len(anchors)//2):
            final.append([anchors[i * 2], anchors[i * 2 + 1]])
        return final

    def __del__(self):
        self.log_writer.close()

if __name__ == "__main__":

    # classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "mouse", "microbit", "ruler", "cat", "peer", "ship", "apple", "car", "pan", "dog", "umbrella", "airplane", "clock", "grape", "cup", "left", "right", "front", "stop", "back"]
    # anchors = [[2.44, 2.25], [5.03, 4.91], [3.5, 3.53], [4.16, 3.94], [2.97, 2.84]]
    # dataset_name = "cards2"

    classes = ["right", "left", "back", "front", "others"]
    anchors = [[1.87, 5.32], [1.62, 3.28], [1.75, 3.78], [1.33, 3.66], [1.5, 4.51]]
    dataset_name = "lobster_5classes"

    train = Train(classes,
                "yolov2_slim",
                dataset_name,
                batch_size=32,
                anchors=anchors,
                input_shape=(3, 224, 224))
    # dataset = Cards_Generator(classes, "datasets/cards/card_in", "datasets/cards/bg", 300, 
    #                     transform = SSDAugmentation(size=(224, 224), mean=(0.5, 0.5, 0.5), std=(128/255.0, 128/255.0, 128/255.0))
    #                     )
    # train.load_dataset("datasets/cards", load_num_workers = 16, dataset=dataset)
    train.load_dataset(f"datasets/{dataset_name}", load_num_workers = 16)
    # train.preview_data()
    train.train(260, eval_every_epoch = 5, save_every_epoch = 5,
                    # resume = "out/lobster_5classes/yolov2_slim/weights/epoch_60.pth"
                    )
    train.report()

