# -*- coding: utf-8 -*-
from imgaug import augmenters as iaa
import cv2
import numpy as np
np.random.seed(1337)
import cv2, random, imutils, os

# bg_folder = "datasets/cards/bg"
# bgs = []
# for bg in os.listdir(bg_folder):
#     if bg.endswith(".jpg"):
#         path = os.path.join(bg_folder, bg)
#         bgs.append(cv2.imread(path))

# imgs = {}
# # labels = ["green", "blue", "yellow", "black", "red", "white"]
# labels = ["0", "1"]
# # labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "back", "forward", "left", "right", "stop", 
# #                                          "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
# #                                          "circle", "hexagon", "pentagon", "rectangle", "quad", "trangle",
# #                                          "airplane", "apple", "ship", "bread", "car", "cat", "cup", "dog", "egg", "grape", "pear", "strawberry", "umbrella"]
# for label in labels:
#     template_path = "datasets/cards/card_in/{}.png".format(label)
#     img = cv2.imread(template_path)
#     if type(img) == type(None):
#         print("read template error:", template_path)
#         continue
#     imgs[label] = img[3:-4, 4:-5, :] # 这里截取掉了边框的部分,因为拿到的设计图有黑边
# print("img templates:", len(imgs))
# print("img templates keys:", imgs.keys())

# images_boxs = []

# def paste(l_img, s_img, x_offset, y_offset):
#     y1, y2 = y_offset, y_offset + s_img.shape[0]
#     x1, x2 = x_offset, x_offset + s_img.shape[1]

#     alpha_s = s_img[:, :, 3] / 255.0
#     alpha_l = 1.0 - alpha_s

#     for c in range(0, 3):
#         l_img[y1:y2, x1:x2, c] = (alpha_s * s_img[:, :, c] +
#                                   alpha_l * l_img[y1:y2, x1:x2, c])
#     return l_img

# def one_in_another(pic1, pic2, pic3, th_area_diff):
#     # 两个图像相交， 并且面积和与pic3的面积近似，就认为实际上是一个图形
#     if ( (pic1[0] + pic1[2] < pic2[0] or pic2[0] + pic2[2] < pic1[0] )# x 轴不相交
#            and (pic1[1] + pic1[3] < pic2[1] or pic2[1] + pic2[3] < pic1[1])# y轴不相交
#        ):
#         return False
#     if abs(pic1[2] * pic1[3] + pic2[2] * pic2[3] - pic3[2] * pic3[3]) < th_area_diff:
#         return True
#     return False
    

# def find_card(pic_name, canny_th1, canny_th2, th_length = 300, th_points_min = 4, th_points_max = 10, show=False):
#     if type(pic_name) ==  str:
#         print("find card in {}".format(pic_name))
#         img = cv2.imread(pic_name)
#     else:
#         img = pic_name
#     edged = cv2.Canny(img, canny_th1, canny_th2)
#     newimage=img.copy()
#     final_img = img.copy()
    
#     th_rel_w_h =  img.shape[0] / 4

#     contours = cv2.findContours(edged, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
#     cnts = contours[0]
# #     print(cnts[0].shape)
#     if len(cnts) > 0:
#         points = []
#         for c in cnts:
#             for p in c:
#                 points.append(p)
#         points = np.array(points)
# #         print(points.shape)
#         points_f = points.reshape((points.shape[0], points.shape[2])).astype('float32')
#         criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
#         flags = cv2.KMEANS_RANDOM_CENTERS
#         compactness,labels,centers  = cv2.kmeans(points_f, 2, None, criteria, 10, flags)
# #         print(compactness,centers)
#         dis = np.sqrt( np.square(centers[0][0] - centers[1][0]) + np.square(centers[0][1] - centers[1][1]) )
# #         print("distance:", dis)
#         for i, p in enumerate(points):
#             cv2.circle(newimage, (p[0][0], p[0][1]), 2, (0, 0, 255) if labels[i]==0 else (0, 255, 0), -1)
#         p2 = [[], []]
#         for i, p in enumerate(points):
#             if labels[i] == 0:
#                 p2[0].append(p)
#             else:
#                 p2[1].append(p)
#         p2 = [np.array(p2[0]), np.array(p2[1])]
#         x, y, w, h = cv2.boundingRect(p2[0])
#         cv2.rectangle(newimage, (x, y), (x+w, y+h), (0, 255, 0 ), 2)
#         x2, y2, w2, h2 = cv2.boundingRect(p2[1])
#         cv2.rectangle(newimage, (x2, y2), (x2+w2, y2+h2), (0, 255, 0 ), 2)
#         rel_area = abs(w*h - w2*h2)
#         rel_w = abs(w2-w)
#         rel_h = abs(h2-h)
# #         print("rel_area：{}, rel_width:{}, rel_height;{}".format(rel_area, rel_w, rel_h  ) )
#         x3, y3, w3, h3 = cv2.boundingRect(points)
#         if (rel_w < th_rel_w_h and rel_h < th_rel_w_h) or one_in_another((x,y,w,h), (x2,y2,w2,h2), (x3, y3, w3, h3), np.square(img.shape[0]/12)): #两个区域形状相差不明显，或者其中一个几乎属于另一个框内
#                 cv2.rectangle(final_img, (x3, y3), (x3+w3, y3+h3), (0, 255, 0 ), 2)
#         else:
#                 # 将轮廓按大小降序排序
#                 cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
#                 max_length = th_length
#                 max_length_points = None
#                 for c in cnts:
#                     peri = cv2.arcLength(c, True)
#                     approx = cv2.approxPolyDP(c, 0.02 * peri, True)
#         #             print("arc length:{:.2f}, point length:{}".format(peri, len(approx)))
#                     if len(approx) >= th_points_min and len(approx) <= th_points_max and peri > max_length:
#         #                 print("--")
#                         max_length_points = approx
#                         max_length = peri
#                 if type(max_length_points) == np.ndarray:
#                     x, y, w, h = cv2.boundingRect(max_length_points)
#                     cv2.rectangle(final_img, (x, y), (x+w, y+h), (0, 255, 0 ), 2)

#     if show:
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         edged = cv2.cvtColor(edged, cv2.COLOR_BGR2RGB)
#         newimage = cv2.cvtColor(newimage, cv2.COLOR_BGR2RGB)
#         final_img = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)
#         fig, (ax0,ax1,ax2, ax3) = plt.subplots(1,4,figsize=(9,3))
#         ax0.imshow(img)
#         ax1.imshow(edged)
#         ax2.imshow(newimage)
#         ax3.imshow(final_img)
#     return x3, y3, w3, h3


# def gen_image(template_img):
#     global bgs

#     canny_th1 = 50
#     canny_th2 = 240

#     img = template_img
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

#     w = img.shape[1]
#     h = img.shape[0]
#     j = random.randint(1, 2)
#     x0 = random.randint(0, w//(4*j))
#     x1 = random.randint(0, w//(4*j))
#     x2 = random.randint(w-w//(4*j), w)
#     x3 = random.randint(w-w//(4*j), w)
#     y0 = random.randint(0, h//(4*j))
#     y1 = random.randint(h-h//(4*j), h)
#     y2 = random.randint(h-h//(4*j), h)
#     y3 = random.randint(0, h//(4*j))
#     src = np.array([[0,0], [0,h], [w, h], [w, 0]], dtype=np.float32)
#     dst = np.array([[x0, y0], [x1, y1], [x2, y2], [x3, y3]], dtype=np.float32)
#     M = cv2.getPerspectiveTransform(src, dst)
#     img2 = cv2.warpPerspective(img, M, (w, h))
#     rotate_angle = random.randint(0, 360)
#     img2 = imutils.rotate_bound(img2, rotate_angle)
#     bg_index = random.randint(0, len(bgs)-1)
#     img3 = bgs[bg_index].copy()
#     # limit width and height
#     if img2.shape[0] > img2.shape[1]: # h > w
#         new_h = random.randint(img3.shape[0]//4, img3.shape[0]-10)
#         new_w = int(new_h/img2.shape[0] * img2.shape[1])
#     else:
#         new_w = random.randint(img3.shape[1]//4, img3.shape[1]-10)
#         new_h = int(new_w/img2.shape[1] * img2.shape[0])
#     img2 = cv2.resize(img2, (new_w, new_h))
#     pos_x = random.randint(0, img3.shape[1] - new_w)
#     pos_y = random.randint(0, img3.shape[0] - new_h)
#     img3 = paste(img3, img2, pos_x, pos_y)
#     white_bg = np.zeros(img3.shape, dtype=np.uint8)
#     white_bg[:] = (255,255,255)
#     white_bg = paste(white_bg, img2, pos_x, pos_y)
#     x, y, w, h = find_card(white_bg, th_points_max=6, canny_th1=canny_th1, canny_th2=canny_th2, show=False)
#     del white_bg
#     mask = np.zeros(img3.shape, dtype=np.uint8)
#     contrast = random.randint(5, 12)/10.0
#     brightness = random.randint(0, 10)
#     img3 = cv2.addWeighted(img3, contrast, mask, 0, brightness) 
#     del mask
#     pos = np.array([[x, y, x+w, y+h ]])
#     images_boxs.append((w, h))
# #     print(len(images_boxs))
#     return img3, pos
#     # cv2.rectangle(img3, (x, y), (x+w, y+h), (0, 255, 0 ), 2)
#     # img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
#     # plt.subplot(num1, num2, i+1)
#     # plt.imshow(img3)
#     # plt.show()
class ImgAugment(object):
    def __init__(self, h, w, jitter):
        """
        # Args
            desired_w : int
            desired_h : int
            jitter : bool
        """
        self._jitter = jitter
        self._w = w
        self._h = h
        
    def imread(self, img_file, boxes):
        """
        # Args
            img_file : str
            boxes : array, shape of (N, 4)
        
        # Returns
            image : 3d-array, shape of (h, w, 3)
            boxes_ : array, same shape of boxes
                jittered & resized bounding box
        """
        # 1. read image file
        image = cv2.imread(img_file)
        # datasets/cards/thing/dog/81.jpg
        # label = os.path.split(img_file)[0].split("/")[3]
        # image = imgs[label]
        # image, boxes = gen_image(image)


        if image is None:
            print("Image Path: " + img_file)
            raise ValueError
    
        # 2. make jitter on image
        boxes_ = np.copy(boxes)
        if self._jitter:
            image, boxes_ = make_jitter_on_image(image, boxes_)
    
        # 3. resize image            
        image, boxes_ = resize_image(image, boxes_, self._w, self._h)
        return image, boxes_

    def make_jitter(self, img, boxes):
        if self._jitter:
            img, boxes = make_jitter_on_image(img, boxes)
        return img, boxes

def make_jitter_on_image(image, boxes):
    h, w, _ = image.shape

    ### scale the image
    scale = np.random.uniform() / 10. + 1.
    image = cv2.resize(image, (0,0), fx = scale, fy = scale)

    ### translate the image
    max_offx = (scale-1.) * w
    max_offy = (scale-1.) * h
    offx = int(np.random.uniform() * max_offx)
    offy = int(np.random.uniform() * max_offy)
    
    image = image[offy : (offy + h), offx : (offx + w)]

    ### flip the image
    #flip = np.random.binomial(1, .5)
    #if flip > 0.5:
    #    image = cv2.flip(image, 1)
    #    is_flip = True
    #else:
    #    is_flip = False

    aug_pipe = _create_augment_pipeline()
    image = aug_pipe.augment_image(image)
    
    # fix object's position and size
    new_boxes = []
    for box in boxes:
        x1,y1,x2,y2 = box
        x1 = int(x1 * scale - offx)
        x2 = int(x2 * scale - offx)
        
        y1 = int(y1 * scale - offy)
        y2 = int(y2 * scale - offy)

    #    if is_flip:
    #        xmin = x1
    #        x1 = w - x2
    #        x2 = w - xmin
        new_boxes.append([x1,y1,x2,y2])
    return image, np.array(new_boxes)


def resize_image(image, boxes, desired_w, desired_h):
    h, w, _ = image.shape
    
    # resize the image to standard size
    image = cv2.resize(image, (desired_h, desired_w))
    image = image[:,:,::-1]

    # fix object's position and size
    new_boxes = []
    for box in boxes:
        x1,y1,x2,y2 = box
        x1 = int(x1 * float(desired_w) / w)
        x1 = max(min(x1, desired_w - 1), 0)
        x2 = int(x2 * float(desired_w) / w)
        x2 = max(min(x2, desired_w - 1), 0)
        
        y1 = int(y1 * float(desired_h) / h)
        y1 = max(min(y1, desired_h - 1), 0)
        y2 = int(y2 * float(desired_h) / h)
        y2 = max(min(y2, desired_h - 1), 0)

        new_boxes.append([x1,y1,x2,y2])
    return image, np.array(new_boxes)


def _create_augment_pipeline():
    
    ### augmentors by https://github.com/aleju/imgaug
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)

    # Define our sequence of augmentation steps that will be applied to every image
    # All augmenters with per_channel=0.5 will sample one value _per image_
    # in 50% of all cases. In all other cases they will sample new values
    # _per channel_.
    aug_pipe = iaa.Sequential(
        [
            # apply the following augmenters to most images
            #iaa.Fliplr(0.5), # horizontally flip 50% of all images
            #iaa.Flipud(0.2), # vertically flip 20% of all images
            #sometimes(iaa.Crop(percent=(0, 0.1))), # crop images by 0-10% of their height/width
            #sometimes(iaa.Affine(
                #scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
                #translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
                #rotate=(-5, 5), # rotate by -45 to +45 degrees
                #shear=(-5, 5), # shear by -16 to +16 degrees
                #order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                #cval=(0, 255), # if mode is constant, use a cval between 0 and 255
                #mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
            #)),
            # execute 0 to 5 of the following (less important) augmenters per image
            # don't execute all of them, as that would often be way too strong
            iaa.SomeOf((0, 5),
                [
                    #sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # convert images into their superpixel representation
                    iaa.OneOf([
                        iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0
                        iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
                        iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 2 and 7
                    ]),
                    iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
                    #iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
                    # search either for all edges or for directed edges
                    #sometimes(iaa.OneOf([
                    #    iaa.EdgeDetect(alpha=(0, 0.7)),
                    #    iaa.DirectedEdgeDetect(alpha=(0, 0.7), direction=(0.0, 1.0)),
                    #])),
                    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images
                    iaa.OneOf([
                        iaa.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove up to 10% of the pixels
                        #iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                    ]),
                    #iaa.Invert(0.05, per_channel=True), # invert color channels
                    iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                    iaa.Multiply((0.5, 1.5), per_channel=0.5), # change brightness of images (50-150% of original value)
                    iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5), # improve or worsen the contrast
                    #iaa.Grayscale(alpha=(0.0, 1.0)),
                    #sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), # move pixels locally around (with random strengths)
                    #sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))) # sometimes move parts of the image around
                ],
                random_order=True
            )
        ],
        random_order=True
    )
    return aug_pipe


if __name__ == '__main__':
    
    img_file = "C://Users//penny//git//basic-yolo-keras//sample//raccoon_train_imgs//raccoon-1.jpg"
    objects = [{'name': 'raccoon', 'xmin': 81, 'ymin': 88, 'xmax': 522, 'ymax': 408},
               {'name': 'raccoon', 'xmin': 100, 'ymin': 100, 'xmax': 400, 'ymax': 300}]
    boxes = np.array([[81,88,522,408],
                      [100,100,400,300]])
    
    desired_w = 416
    desired_h = 416
    jitter = True
    
    aug = ImgAugment(desired_h, desired_w, jitter)
    img, boxes_ = aug.imread(img_file, boxes)
    img = img.astype(np.uint8)
    
    import matplotlib.pyplot as plt
    for box in boxes_:
        x1, y1, x2, y2 = box
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 3)
    plt.imshow(img)
    plt.show()

    
    
    



