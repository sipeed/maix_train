# -*- coding: utf-8 -*-

import os
import numpy as np
from xml.etree.ElementTree import parse


def get_unique_labels(files):
    parser = PascalVocXmlParser()
    labels = []
    for fname in files:
        labels += parser.get_labels(fname)
        labels = list(set(labels))
    labels.sort()
    return labels


def get_train_annotations(labels,
                          img_folder = None,
                          ann_folder = None,
                          valid_img_folder = "",
                          valid_ann_folder = "",
                          img_in_mem = None,
                          ann_in_mem = None,
                          valid_img_in_mem = None,
                          valid_ann_in_mem = None,
                          is_only_detect=False,
                          classes=[]):
    """
    # Args
        labels : list of strings
            ["raccoon", "human", ...]
        img_folder : str
        ann_folder : str
        valid_img_folder : str
        valid_ann_folder : str

    # Returns
        train_anns : Annotations instance
        valid_anns : Annotations instance
    """
    # check param
    if (img_in_mem is not None and img_folder):
        raise Exception("param error, arg img_in_mem and img_folder only one of them should be give")

    # parse annotations of the training set
    if img_folder:
        train_anns = parse_annotation(ann_folder,
                                        img_folder,
                                        labels,
                                        is_only_detect,
                                        classes)
        # parse annotations of the validation set, if any, otherwise split the training set
        if os.path.exists(valid_ann_folder):
            valid_anns = parse_annotation(valid_ann_folder,
                                            valid_img_folder,
                                            labels,
                                            is_only_detect,
                                            classes)
        else:
            train_valid_split = int(0.8*len(train_anns))
            train_anns.shuffle()
            
            # Todo : Hard coding
            valid_anns = Annotations(train_anns._label_namings)
            valid_anns._components = train_anns._components[train_valid_split:]
            train_anns._components = train_anns._components[:train_valid_split]
    else:
        train_anns = parse_annotation_in_mem(ann_in_mem, img_in_mem, labels, is_only_detect)
        if valid_ann_in_mem:
            valid_anns = parse_annotation_in_mem(valid_ann_in_mem, valid_img_in_mem, labels, is_only_detect,)
        else:
            train_valid_split = int(0.8*len(train_anns))
            train_anns.shuffle()
            
            # Todo : Hard coding
            valid_anns = Annotations(train_anns._label_namings, img_in_memory=True)
            valid_anns._components = train_anns._components[train_valid_split:]
            train_anns._components = train_anns._components[:train_valid_split]
    
    return train_anns, valid_anns


class PascalVocXmlParser(object):
    """Parse annotation for 1-annotation file """
    
    def __init__(self):
        pass

    def get_fname(self, annotation_file):
        """
        # Args
            annotation_file : str
                annotation file including directory path
        
        # Returns
            filename : str
        """
        root = self._root_tag(annotation_file)
        return root.find("filename").text

    def get_width(self, annotation_file):
        """
        # Args
            annotation_file : str
                annotation file including directory path
        
        # Returns
            width : int
        """
        tree = self._tree(annotation_file)
        for elem in tree.iter():
            if 'width' in elem.tag:
                return int(elem.text)

    def get_height(self, annotation_file):
        """
        # Args
            annotation_file : str
                annotation file including directory path
        
        # Returns
            height : int
        """
        tree = self._tree(annotation_file)
        for elem in tree.iter():
            if 'height' in elem.tag:
                return int(elem.text)

    def get_labels(self, annotation_file):
        """
        # Args
            annotation_file : str
                annotation file including directory path
        
        # Returns
            labels : list of strs
        """

        root = self._root_tag(annotation_file)
        labels = []
        obj_tags = root.findall("object")
        for t in obj_tags:
            labels.append(t.find("name").text)
        return labels
    
    def get_boxes(self, annotation_file):
        """
        # Args
            annotation_file : str
                annotation file including directory path
        
        # Returns
            bbs : 2d-array, shape of (N, 4)
                (x1, y1, x2, y2)-ordered
        """
        root = self._root_tag(annotation_file)
        bbs = []
        obj_tags = root.findall("object")
        for t in obj_tags:
            box_tag = t.find("bndbox")
            x1 = box_tag.find("xmin").text
            y1 = box_tag.find("ymin").text
            x2 = box_tag.find("xmax").text
            y2 = box_tag.find("ymax").text
            box = np.array([int(float(x1)), int(float(y1)), int(float(x2)), int(float(y2))])
            bbs.append(box)
        bbs = np.array(bbs)
        return bbs

    def _root_tag(self, fname):
        tree = parse(fname)
        root = tree.getroot()
        return root

    def _tree(self, fname):
        tree = parse(fname)
        return tree

def parse_annotation(ann_dir, img_dir, labels_naming=[], is_only_detect=False, classes=[]):
    """
    # Args
        ann_dir : str
        img_dir : str
        labels_naming : list of strings
    
    # Returns
        all_imgs : list of dict
    """
    parser = PascalVocXmlParser()
    
    if is_only_detect:
        annotations = Annotations(["object"])
    else:
        annotations = Annotations(labels_naming)
    if len(classes) == 0:
        dirs = os.listdir(ann_dir)
        print(dirs)
        for name in dirs:
            if os.path.isdir(os.path.join(ann_dir, name)):
                classes.append(os.path.basename(name))
    print("class folders: ", classes)
    for cls in classes:
      subclasses = os.listdir(ann_dir+'/'+cls)
      for subcls in subclasses:
        xml_dir = ann_dir+'/'+cls+'/'+subcls
        img_dir_ = img_dir+'/'+cls+'/'+subcls
        if not os.path.isdir(xml_dir) or not os.path.isdir(img_dir_):
            continue
#         files_limit = 300
#         count = 0
        for ann in sorted(os.listdir(xml_dir)):

          if os.path.isdir(ann) or  not ann.endswith(".xml"):
            continue
#           if count > files_limit:
#             break
#           count += 1
          annotation_file = os.path.join(xml_dir, ann)
          fname = parser.get_fname(annotation_file)

          annotation = Annotation(os.path.join(img_dir_, fname))

          labels = parser.get_labels(annotation_file)
          boxes = parser.get_boxes(annotation_file)
        
          for label, box in zip(labels, boxes):
            x1, y1, x2, y2 = box
            if is_only_detect:
                annotation.add_object(x1, y1, x2, y2, name="object")
            else:
                if label in labels_naming:
                    annotation.add_object(x1, y1, x2, y2, name=label)
                    
          if annotation.boxes is not None:
            annotations.add(annotation)
                        
    return annotations

def parse_annotation_in_mem(anns, imgs, labels_naming=[], is_only_detect=False, classes=[]):
    """
    # Args
        ann_in_mem : list [ [[xmin, ymin, xmax, ymax],], ]
        img_in_mem : list
        labels_naming : list of strings
    
    # Returns
        all_imgs : list of dict
    """
    
    if is_only_detect:
        annotations = Annotations(["object"], img_in_memory = True)
    else:
        annotations = Annotations(labels_naming, img_in_memory = True)
    for i, img in enumerate(imgs):
        annotation = Annotation(img = img)
        for bbox in anns[i]:
            if is_only_detect:
                annotation.add_object(bbox[0], bbox[1], bbox[2], bbox[3], name="object")
            else:
                # if label in labels_naming:
                annotation.add_object(bbox[0], bbox[1], bbox[2], bbox[3], name=labels_naming[bbox[4]])
        if annotation.boxes is not None:
            annotations.add(annotation)
    return annotations
            

class Annotation(object):
    """
    # Attributes
        fname : image file path
        labels : list of strings
        boxes : Boxes instance
    """
    def __init__(self, filename = None, img = None):
        self.fname = filename
        self.img = img
        self.labels = []
        self.boxes = None

    def add_object(self, x1, y1, x2, y2, name):
        self.labels.append(name)
        if self.boxes is None:
            self.boxes = np.array([x1, y1, x2, y2]).reshape(-1,4)
        else:
            box = np.array([x1, y1, x2, y2]).reshape(-1,4)
            self.boxes = np.concatenate([self.boxes, box])

class Annotations(object):
    def __init__(self, label_namings, img_in_memory=False):
        self._components = []
        self._label_namings = label_namings
        self.is_img_in_memory = img_in_memory

    def n_classes(self):
        return len(self._label_namings)

    def add(self, annotation):
        self._components.append(annotation)

    def shuffle(self):
        np.random.shuffle(self._components)
    
    def fname(self, i):
        index = self._valid_index(i)
        return self._components[index].fname
    
    def img(self, i):
        index = self._valid_index(i)
        return self._components[index].img
    
    def boxes(self, i):
        index = self._valid_index(i)
        return self._components[index].boxes

    def labels(self, i):
        """
        # Returns
            labels : list of strings
        """
        index = self._valid_index(i)
        return self._components[index].labels

    def code_labels(self, i):
        """
        # Returns
            code_labels : list of int
        """
        str_labels = self.labels(i)
        labels = []
        for label in str_labels:
            labels.append(self._label_namings.index(label))
        return labels

    def _valid_index(self, i):
        valid_index = i % len(self._components)
        return valid_index

    def __len__(self):
        return len(self._components)

    def __getitem__(self, idx):
        return self._components[idx]

