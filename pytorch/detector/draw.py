import cv2, random

class Draw:
    def __init__(self, classes):
        self.draw_init(classes)

    def draw_img(self, img, taget, labels, class_names, scores = None, thresh = -1):
        h, w, c = img.shape
        img = cv2.UMat(img)
        for i, gt in enumerate(taget):
            xmin, ymin, xmax, ymax = (int(gt[0]*w), int(gt[1]*h), int(gt[2]*w), int(gt[3]*h))
            label = int(labels[i])

            if (not scores is None) and thresh > 0 and scores[i] < thresh:
                continue
            cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), self.bgs_color[label], 1)
            cv2.rectangle(img, (int(xmin), int(abs(ymin)-20)), (int(xmax), int(ymin)), self.bgs_color[label], -1)
            if not scores is None:
                mess = '%s, %.2f' % (class_names[int(label)], scores[i])
            else:
                mess = '%s' % (class_names[int(label)])
            cv2.putText(img, mess, (int(xmin), int(ymin-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.fonts_color[label], 1)
        return img

    def draw_init(self, classes):
        self.bgs_color = []
        self.fonts_color = []
        for label in classes:
            bg = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            color = (0, 0, 0) if (bg[0] * 0.299 + bg[1] * 0.587 + bg[2] * 0.114) > 127 else (256, 256, 256)
            self.bgs_color.append(bg)
            self.fonts_color.append(color)


