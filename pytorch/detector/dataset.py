import os
import xml.etree.ElementTree as ET
import cv2
import numpy as np
from glob import glob

try:
    from .logger import Fake_Logger
except Exception:
    from logger import Fake_Logger

class CustomAnnotationParser(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, classes, class_to_ind=None, keep_difficult=False, log=Fake_Logger()):
        self.log = log
        self.class_to_ind = class_to_ind or dict(
            zip(classes, range(len(classes))))
        self.keep_difficult = keep_difficult

    def __call__(self, path, width=0, height=0, ignore_other_obj = False, return_size = False):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        if not os.path.exists(path):
            if return_size:
                return None, None
            return None
        target = ET.parse(path).getroot()
        res = []
        if width <= 0 or height <= 0:
            size_obj = list(target.iter('size'))[0]
            width = int(size_obj.find("width").text)
            height = int(size_obj.find("height").text)
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.strip()
            bbox = obj.find('bndbox')
            if not name in self.class_to_ind:
                if not ignore_other_obj:
                    self.log.w(f"obj {name} error: {path}, not in classes: {self.class_to_ind}") # target.find('path').text)
                continue
            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                # scale height or width
                cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]
            # img_id = target.find('filename').text[:-4]
        if return_size:
            return res, (width, height)
        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


class Dataset_VOC:
    def __init__(self, classes, path, sets=["train"], check_datasets=True,
                transform = None,
                log = Fake_Logger(),
                min_box_size = (10, 10)):
        self.min_box_size = min_box_size
        self.log = log
        self.classes = classes
        self.root = path
        self.transform = transform
        self.target_parser = CustomAnnotationParser(self.classes, log=self.log)
        self._annopath = os.path.join('%s', 'Annotations', '%s.xml')
        self._imgpath = os.path.join('%s', 'JPEGImages', '%s.jpg')
        self.ids = []
        for name in sets:
            self.log.i(f"check dataset in {name}")
            with open(os.path.join(self.root, 'ImageSets', 'Main', name + '.txt')) as f:
                lines = f.readlines()
            for i, line in enumerate(lines):
                line = line.strip()
                item = (self.root, line)
                if os.path.exists(self._annopath % item):
                    if check_datasets:
                        ok = self.pull_item(0, item, test = True)
                        if not ok:
                            continue
                    self.ids.append(item)
                else:
                    log.w("file not found: ", self._annopath % (self.root, line))
                if i % int(len(lines) * 0.05) == 0:
                    print(f"{i}/{len(lines)}({i/len(lines)*100:.01f}%)", end=", ", flush=True)
            print("")

    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)
        if type(gt) == type(None):
            self.log.e("please trun on check_datasets arg of Dataset_VOC to see error")
            raise Exception()
        return im, gt

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index, img_id = None, test = False):
        if test:
            if not img_id:
                img_id = self.ids[index]
            anno_path = self._annopath % img_id
            target, (w, h) = self.target_parser(anno_path, ignore_other_obj=True, return_size=True)
            final_target = []
            for box in target:
                if ((box[2] - box[0]) * w < self.min_box_size[0]) or ((box[3] - box[1]) * h < self.min_box_size[1]):
                    self.log.w("ignore small box of {}".format(img_id[1]))
                    continue
                final_target.append(box)
            return final_target
        else:
            img_id = self.ids[index]
            img = cv2.imread(self._imgpath % img_id)
            height, width, channels = img.shape
            anno_path = self._annopath % img_id
            target = self.target_parser(anno_path, width, height)
            if not target:
                return None, None, 0, 0

            if self.transform is not None :
                target = np.array(target)
                img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
                # to rgb
                img = img[:, :, (2, 1, 0)]
                # img = img.transpose(2, 0, 1)
                target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
            return img.transpose(2, 0, 1), target, height, width

    def pull_item_raw(self, index, annotation = True):
        img_id = self.ids[index]
        img = cv2.imread(self._imgpath % img_id)
        if annotation:
            height, width, channels = img.shape
            anno_path = self._annopath % img_id
            target = self.target_parser(anno_path, width, height)
            if not target:
                return None, None
            return img, target
        return img

    def pull_item2(self, index, denorm):
        img, target , w, h = self.pull_item(index)
        img = img.transpose(1, 2, 0)
        img = img[:, :, (2, 1, 0)]
        img = denorm(img)
        return img, target

    def pull_image(self, index, get_img_path = False):
        img_id = self.ids[index]
        img = cv2.imread(self._imgpath % img_id)
        if get_img_path:
            return img, self._imgpath % img_id, f'{img_id[1]}.jpg'
        return img

class Dataset_Folder:
    def __init__(self, path, transform = None, log = Fake_Logger()):
        self.log = log
        self.root = path
        self.transform = transform
        self.ids = []
        pwd = os.getcwd()
        os.chdir(self.root)
        files = glob("**/*.*")
        files.extend(glob("*.*"))
        os.chdir(pwd)
        for name in files:
            ext = os.path.splitext(name)[-1].lower()
            if ext == ".jpg" or ext == ".jpeg" or ext == ".png":
                img_path = os.path.join(self.root, name)
                img = cv2.imread(img_path)
                if img is None:
                    self.log.w("image {} open fail!".format(img_path))
                    continue
                self.ids.append((img_path, name))
                

    def __getitem__(self, index):
        img = cv2.imread(self.ids[index][0])
        if img is None:
            self.log.w("read img {} fail".format(self.ids[index][0]))
            return None
        if self.transform is not None :
            img = self.transform(img)
            # to rgb
            img = img[:, :, (2, 1, 0)]
        return img.transpose(2, 0, 1)

    def __len__(self):
        return len(self.ids)

    def pull_image(self, index, get_img_path = False):
        '''
            @return if get_img_path: img, path_abs, path_relative
                    else:            img
        '''
        img = cv2.imread(self.ids[index][0])
        if img is None:
            self.log.w("read img {} fail".format(self.ids[index][0]))
            return None
        if get_img_path:
            return img, self.ids[index][0], self.ids[index][1]
        return img

