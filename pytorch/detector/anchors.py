
import numpy as np
try:
    from .logger import Fake_Logger, Logger
    from .dataset import Dataset_VOC
    from .augmentations import SSDAugmentation
except Exception:
    from logger import Fake_Logger, Logger
    from dataset import Dataset_VOC
    from augmentations import SSDAugmentation



def iou(box, clusters):
    """
    Calculates the Intersection over Union (IoU) between a box and k clusters.
    :param box: tuple or array, shifted to the origin (i. e. width and height)
    :param clusters: numpy array of shape (k, 2) where k is the number of clusters
    :return: numpy array of shape (k, 0) where k is the number of clusters
    """
    x = np.minimum(clusters[:, 0], box[0])
    y = np.minimum(clusters[:, 1], box[1])
    if np.count_nonzero(x == 0) > 0 or np.count_nonzero(y == 0) > 0:
        raise ValueError("Box has no area")

    intersection = x * y
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]

    iou_ = intersection / (box_area + cluster_area - intersection)

    return iou_


def avg_iou(boxes, clusters):
    """
    Calculates the average Intersection over Union (IoU) between a numpy array of boxes and k clusters.
    :param boxes: numpy array of shape (r, 2), where r is the number of rows
    :param clusters: numpy array of shape (k, 2) where k is the number of clusters
    :return: average IoU as a single float
    """
    return np.mean([np.max(iou(boxes[i], clusters)) for i in range(boxes.shape[0])])


def translate_boxes(boxes):
    """
    Translates all the boxes to the origin.
    :param boxes: numpy array of shape (r, 4)
    :return: numpy array of shape (r, 2)
    """
    new_boxes = boxes.copy()
    for row in range(new_boxes.shape[0]):
        new_boxes[row][2] = np.abs(new_boxes[row][2] - new_boxes[row][0])
        new_boxes[row][3] = np.abs(new_boxes[row][3] - new_boxes[row][1])
    return np.delete(new_boxes, [0, 1], axis=1)


def kmeans(boxes, k, dist=np.median):
    """
    Calculates k-means clustering with the Intersection over Union (IoU) metric.
    :param boxes: numpy array of shape (r, 2), where r is the number of rows
    :param k: number of clusters
    :param dist: distance function
    :return: numpy array of shape (k, 2)
    """
    rows = boxes.shape[0]

    distances = np.empty((rows, k))
    last_clusters = np.zeros((rows,))

    np.random.seed()

    # the Forgy method will fail if the whole array contains the same rows
    clusters = boxes[np.random.choice(rows, k, replace=False)]

    while True:
        for row in range(rows):
            distances[row] = 1 - iou(boxes[row], clusters)

        nearest_clusters = np.argmin(distances, axis=1)

        if (last_clusters == nearest_clusters).all():
            break

        for cluster in range(k):
            d = boxes[nearest_clusters == cluster]
            if len(d) == 0:
                continue
            clusters[cluster] = dist(d, axis=0)
        last_clusters = nearest_clusters

    return clusters


class Anchors:
    def __init__(self, log = Fake_Logger()):
        self.log = log

    def get_boxes_by_dataset(self, dataset):
        boxes = []
        for i in range(len(dataset)):
            target = dataset.pull_item(i, test = True)
            for bbox in target:
                boxes.append(( bbox[2] - bbox[0], bbox[3] - bbox[1]))
        return boxes

    def get_anchors(self, dataset = None, bboxes = None, net_in_size=(224, 224), clusters = 5, net_out_size=(7, 7)):
        '''
            @net_in_size tuple (w, h)
            @bboxes_in format: [ [xmin,ymin, xmax, ymax], ]
                        value range: x [0, w], y [0, h]
            @return anchors, format: list, item is rectangle list, [ [w0, h0], [w1, h1], ...]
        '''
        if not dataset and not bboxes:
            raise ValueError("param datasets or bboxes is needed")
        w = net_in_size[0]
        h = net_in_size[1]
        if dataset:
            bboxes = self.get_boxes_by_dataset(dataset)
        bboxes = np.array(bboxes)
        self.log.i(f"bboxes num: {len(bboxes)}, first bbox: {bboxes[0]}")
        out = kmeans(bboxes, k=clusters)
        iou = avg_iou(bboxes, out) * 100
        self.log.i("bbox accuracy(IOU): {:.2f}%".format(iou))
        self.log.i("bound boxes: {}".format( ",".join("({:.0f},{:.0f})".format(item[0] * w, item[1] * h) for item in out) ))
        for i, wh in enumerate(out):
            out[i][0] = wh[0]*net_out_size[0]
            out[i][1] = wh[1]*net_out_size[1]
        anchors = list(out.flatten())
        ratios = np.around(out[:, 0] / out[:, 1], decimals=2).tolist()
        self.log.i("w/h ratios: {}".format(sorted(ratios)))
        final_anchors = []
        for i in range(0, len(anchors) // 2):
            final_anchors.append([round(anchors[i * 2], 2), round(anchors[i * 2 + 1], 2)])
        self.log.i(f"anchors: {final_anchors}")
        return final_anchors

if __name__ == "__main__":
    # classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "mouse", "microbit", "ruler", "cat", "peer", "ship", "apple", "car", "pan", "dog", "umbrella", "airplane", "clock", "grape", "cup", "left", "right", "front", "stop", "back"]
    # path = "datasets/cards"

    classes = ["right", "left", "back", "front", "others"]
    path = "datasets/lobster_5classes"

    log = Logger()
    input_size = (224, 224) # w, h
    dataset = Dataset_VOC(classes, path, sets=["train"], log = log ,
                               transform = SSDAugmentation(size=input_size, mean=(0.5, 0.5, 0.5), std=(128/255.0, 128/255.0, 128/255.0))
                               )
    a = Anchors()
    anchors = a.get_anchors(dataset=dataset, net_in_size=(224, 224), clusters=5, net_out_size=(7, 7))
    print("anchors: ", anchors)
