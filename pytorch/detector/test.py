import os, sys
import cv2
from collections import Iterable
try:
    from .dataset import Dataset_Folder
    from .dataloader import DataLoader
    from .logger import Logger
    from .augmentations import SSDAugmentationTest, DeNormalize
    from .draw import Draw
except Exception:
    from dataset import Dataset_Folder, Dataset_VOC
    from dataloader import DataLoader
    from logger import Logger
    from augmentations import SSDAugmentationTest, DeNormalize
    from draw import Draw

class Net_Test:
    def __init__(self, dataset, classes, net_type, saved_state_path, input_shape=(3, 416, 416), anchors = None, temp_dir=None, conf_thresh=0.3, nms_thresh=0.3, opt = {}, log = Logger(), device="cuda"):
        '''
            @input_layout only support default now, pytorch is chw, tensorflow is hwc
        '''
        self.classes = classes
        self.net_type = net_type
        self.log = log
        self.anchors = self.val_anchors(anchors)
        self.input_shape = input_shape
        if not temp_dir:
            temp_dir = os.path.join("out", net_type, "test_result")
        self.temp_dir = temp_dir
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)
        self.root = os.path.abspath(os.path.dirname(__file__))
        detectors_path = os.path.join(self.root, "detectors")
        sys.path.insert(0, detectors_path)
        self.detector = __import__(net_type)
        try:
            self.framework = self.detector.framwork
        except Exception:
            self.framework = "torch"
        self.tester = self.detector.Test(
                            self.classes, self.anchors,
                            (self.input_shape[2], self.input_shape[1]),
                            saved_state_path,
                            self.log,
                            conf_thresh,
                            nms_thresh,
                            device
                        )
        self.dataset = dataset
        self.curr_idx = 0
        self.draw = Draw(self.classes)
    
    def detect(self, index = -1, get_img_path = False):
        if index >= 0:
            img = self.dataset[index]
            img_raw = self.dataset.pull_image(index, get_img_path=get_img_path)
        else:    
            if self.curr_idx >= len(self.dataset):
                return None
            img = self.dataset[self.curr_idx] # img or (img, target)
            if isinstance(img, Iterable):
                img = img[0]
            img_raw = self.dataset.pull_image(self.curr_idx, get_img_path=get_img_path)
            self.curr_idx += 1
        boxes, probs, inds = self.tester.detect(img)
        return img_raw, boxes, probs, inds

    def show(self, img, boxes, probs, inds, save_path = None, threshold = -1):
        img = self.draw.draw_img(img, boxes, inds, self.classes, probs, threshold)
        if save_path:
            cv2.imwrite(save_path, img)
        

    def reset(self):
        self.curr_idx = 0

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

if __name__ == "__main__":
    # classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "mouse", "microbit", "ruler", "cat", "peer", "ship", "apple", "car", "pan", "dog", "umbrella", "airplane", "clock", "grape", "cup", "left", "right", "front", "stop", "back"]
    # anchors = [[2.44, 2.25], [5.03, 4.91], [3.5, 3.53], [4.16, 3.94], [2.97, 2.84]]
    # param_path = "out/yolov2_slim/weights/epoch_460.pth"
    # test_dir = "datasets/cards/cap/left"


    classes = ["right", "left", "back", "front", "others"]
    anchors = [[1.87, 5.32], [1.62, 3.28], [1.75, 3.78], [1.33, 3.66], [1.5, 4.51]]
    param_path = "out/lobster_5classes/yolov2_slim/weights/epoch_15.pth"
    test_dir = "datasets/lobster_5classes"
    is_val_data = True

    input_shape=(3, 224, 224)

    log = Logger()

    if is_val_data:
        dataset = Dataset_VOC(classes, test_dir, sets=["val"], log = log,
                            transform = SSDAugmentationTest(size=input_shape[1:][::-1], mean=(0.5, 0.5, 0.5), std=(128/255.0, 128/255.0, 128/255.0))
                            )
    else:
        dataset = Dataset_Folder(test_dir,
                    transform = SSDAugmentationTest(size=input_shape[1:][::-1], mean=(0.5, 0.5, 0.5), std=(128/255.0, 128/255.0, 128/255.0)),
                    log = log
                    )
    test = Net_Test(dataset,
                classes,
                "yolov2_slim",
                param_path,
                input_shape=input_shape,
                anchors=anchors,
                conf_thresh=0.2,
                nms_thresh=0.3,
                log = log,
                device="cpu"
                )
    count = 0
    while 1:
        result = test.detect()
        if not result:
            break
        img, boxes, probs, inds = result
        out_jpg = "out/test.jpg"
        test.show(img, boxes, probs, inds, save_path=out_jpg)
        input(f"[{count}] see {out_jpg}, press any key to continue")
        count += 1
        

