from .net import SlimYOLOv2
import torch

class Test():
    def __init__(self, classes, anchors, input_size, saved_state_path, log, conf_thresh=0.3, nms_thresh=0.3, device="cuda"):
        self.classes = classes
        self.log = log
        self.input_size = input_size
        self.device = device
        if not anchors:
            log.e("anchors not valid: {}, should be [[w, h], ]".format(anchors))
            raise ValueError()
        self.anchors = anchors
        hr = True if input_size[0] > 400 else False
        self.device = device
        self.net = SlimYOLOv2(device, input_size=input_size, num_classes=len(classes), trainable=False, anchors=anchors, hr=hr, conf_thresh=conf_thresh, nms_thresh=nms_thresh)
        self.net.load_state_dict(torch.load(saved_state_path, map_location=device))
        self.net.to(self.device)

    def detect(self, img):
        with torch.no_grad():
            bboxes, scores, cls_inds = self.net(torch.Tensor(img).unsqueeze(0).to(self.device))
            return bboxes, scores, cls_inds
