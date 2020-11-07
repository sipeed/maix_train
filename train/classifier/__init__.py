'''
    train detector

    @author neucrack@sipeed
    @license Apache 2.0 © 2020 Sipeed Ltd
'''



import sys, os
curr_file_dir = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
#     import os, sys
#     root_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
#     sys.path.append(root_path)
from train_base import Train_Base
from utils import gpu_utils, isascii
from utils.logger import Logger, Fake_Logger

import tempfile
import shutil
import zipfile
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import itertools
import random



class Classifier(Train_Base):
    def __init__(self, input_shape=(224, 224, 3), datasets_dir=None, datasets_zip=None, unpack_dir=None, logger = None,
                max_classes_num=15, min_images_num=40, max_images_num=2000, allow_reshape = False):
        '''
            input_shape: input shape (height, width)
            min_images_num: min image number in one class
        '''
        # import_libs() # 针对多进程
        import tensorflow as tf
        self.input_shape = input_shape
        self.need_rm_datasets = False
        self.datasets_rm_dir = None
        self.model = None
        self.history = None
        self.warning_msg = [] # append warning message here
        if logger:
            self.log = logger
        else:
            self.log = Fake_Logger()
        # unzip datasets
        if datasets_zip:
            self.datasets_dir = self._unpack_datasets(datasets_zip, unpack_dir)
            if not self.datasets_dir:
                self.log.e("can't detect datasets, check zip format")
                raise Exception("can't detect datasets, check zip format")
        elif datasets_dir:
            self.datasets_dir = datasets_dir
        else:
            self.log.e("no datasets args")
            raise Exception("no datasets args")
        # get labels by directory name
        self.labels = self._get_labels(self.datasets_dir)
        # check label
        ok, err_msg = self._is_label_data_valid(self.labels, max_classes_num=max_classes_num, min_images_num=min_images_num, max_images_num=max_images_num)
        if not ok:
            self.log.e(err_msg)
            raise Exception(err_msg)
        # check datasets format
        ok, err_msg = self._is_datasets_shape_valid(self.datasets_dir, self.input_shape)
        if not ok:
            if not allow_reshape:
                self.log.e(err_msg)
                raise Exception(err_msg)
            self.on_warning_message(err_msg)
            
        class _Train_progress_cb(tf.keras.callbacks.Callback):#剩余训练时间回调
            def __init__(self, epochs, user_progress_callback, logger):
                self.epochs = epochs
                self.logger = logger
                self.user_progress_callback = user_progress_callback

            def on_epoch_begin(self, epoch, logs=None):
                self.logger.i("epoch {} start".format(epoch))

            def on_epoch_end(self, epoch, logs=None):
                self.logger.i("epoch {} end: {}".format(epoch, logs))
                if self.user_progress_callback:
                    self.user_progress_callback((epoch + 1) / self.epochs * 100, "train epoch end")

            def on_train_begin(self, logs=None):
                self.logger.i("train start")
                if self.user_progress_callback:
                    self.user_progress_callback(0, "train start")

            def on_train_end(self, logs=None):
                self.logger.i("train end")
                if self.user_progress_callback:
                    self.user_progress_callback(100, "train end")
        self.Train_progress_cb = _Train_progress_cb

    def __del__(self):
        if self.need_rm_datasets:
            try:
                shutil.rmtree(self.datasets_dir)
            except Exception as e:
                try:
                    self.log.e("clean temp files error:{}".format(e))
                except Exception:
                    print("log object invalid")
                
    def train(self, epochs= 100,
                    progress_cb=None,
                    weights=os.path.join(curr_file_dir, "weights", "mobilenet_7_5_224_tf_no_top.h5"),
                    batch_size = 5
                    ):
        self.log.i("train, labels:{}".format(self.labels))
        self.log.d("train, datasets dir:{}".format(self.datasets_dir))
        
        from mobilenet_sipeed import mobilenet
        import tensorflow as tf

        # pooling='avg', use around padding instead padding bottom and right for k210
        base_model = mobilenet.MobileNet0(input_shape=self.input_shape,
                     alpha = 0.75, depth_multiplier = 1, dropout = 0.001, pooling='avg',
                     weights=weights, include_top=False)
        # update top layer
        out = base_model.output
        out = tf.keras.layers.Dropout(0.001, name='dropout')(out)
        preds=tf.keras.layers.Dense(len(self.labels), activation='softmax')(out)
        self.model=tf.keras.models.Model(inputs=base_model.input,outputs=preds)
        # only train top layers
        for layer in self.model.layers[:86]:
            layer.trainable=False
        for layer in self.model.layers[86:]:
            layer.trainable=True
        # #model.compile(loss=tf.keras.losses.categorical_crossentropy,optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),metrics=['accuracy'])
        self.model.compile(optimizer=tf.keras.optimizers.SGD(lr=1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        # #model.compile(optimizer=tf.compat.v1.train.RMSPropOptimizer(learning_rate=1e-3), loss='categorical_crossentropy',metrics=['accuracy'])
        # print model summary
        self.model.summary()

        # train
        # datasets process
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        from tensorflow.keras.applications.mobilenet import preprocess_input
        train_gen = ImageDataGenerator(
                preprocessing_function=preprocess_input,
                rotation_range=180,
                featurewise_center=True,
                featurewise_std_normalization=True,
                width_shift_range=0.2,height_shift_range=0.2,
                zoom_range=0.5,
                shear_range=0.5,
                validation_split=0.2
            )

        train_data = train_gen.flow_from_directory(self.datasets_dir,
                target_size=(self.input_shape[0], self.input_shape[1]),
                color_mode='rgb',
                batch_size=batch_size,
                class_mode='sparse', # None / sparse / binary / categorical
                shuffle=True,
                subset= "training"
                )
        valid_data = train_gen.flow_from_directory(self.datasets_dir,
                target_size=(self.input_shape[0], self.input_shape[1]),
                color_mode='rgb',
                batch_size=batch_size,
                class_mode='sparse',
                shuffle=False,
                subset= "validation"
                )
        self.log.i("train data:{}, valid data:{}".format(train_data.samples, valid_data.samples))
        callbacks = [self.Train_progress_cb(epochs, progress_cb, self.log)]
        self.history = self.model.fit_generator(train_data, validation_data=valid_data,
                                 steps_per_epoch=train_data.samples//batch_size,
                                 validation_steps=valid_data.samples//batch_size,
                                 epochs=epochs,callbacks=callbacks)


    
    def report(self, out_path, limit_y_range=None):
        '''
            generate result charts
        '''
        self.log.i("generate report image")
        if not self.history:
            return
        history = self.history

        # set for server with no Tkagg GUI support, use agg(non-GUI backend)
        plt.switch_backend('agg')
        
        fig, axes = plt.subplots(3, 1, constrained_layout=True, figsize = (10, 16), dpi=100)
        if limit_y_range:
            plt.ylim(limit_y_range)

        # acc and val_acc
        # {'loss': [0.5860330664989357, 0.3398533443955177], 'accuracy': [0.70944744, 0.85026735], 'val_loss': [0.4948340670338699, 0.49342870752194096], 'val_accuracy': [0.7, 0.74285716]}
        if "acc" in history.history:
            kws = {
                "acc": "acc",
                "val_acc": "val_acc",
                "loss": "loss",
                "val_loss": "val_loss"
            }
        else:
            kws = {
                "acc": "accuracy",
                "val_acc": "val_accuracy",
                "loss": "loss",
                "val_loss": "val_loss"
            }
        axes[0].plot( history.history[kws['acc']], color='#2886EA', label="train")
        axes[0].plot( history.history[kws['val_acc']], color = '#3FCD6D', label="valid")
        axes[0].set_title('model accuracy')
        axes[0].set_ylabel('accuracy')
        axes[0].set_xlabel('epoch')
        axes[0].locator_params(integer=True)
        axes[0].legend()

        # loss and val_loss
        axes[1].plot( history.history[kws['loss']], color='#2886EA', label="train")
        axes[1].plot( history.history[kws['val_loss']], color = '#3FCD6D', label="valid")
        axes[1].set_title('model loss')
        axes[1].set_ylabel('loss')
        axes[1].set_xlabel('epoch')
        axes[1].locator_params(integer=True)
        axes[1].legend()

        # confusion matrix
        cm, labels_idx = self._get_confusion_matrix()
        axes[2].imshow(cm, interpolation='nearest', cmap = plt.cm.GnBu)
        axes[2].set_title("confusion matrix")
        # axes[2].colorbar()
        num_local = np.array(range(len(labels_idx)))
        axes[2].set_xticks(num_local)
        axes[2].set_xticklabels(labels_idx.keys(), rotation=45)
        axes[2].set_yticks(num_local)
        axes[2].set_yticklabels(labels_idx.keys())

        thresh = cm.max() / 2. # front color black or white according to the background color
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            axes[2].text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment = 'center',
                    color = 'white' if cm[i, j] > thresh else "black")
        axes[2].set_ylabel('True label')
        axes[2].set_xlabel('Predicted label')

        # save to fs
        fig.savefig(out_path)
        plt.close()
        self.log.i("generate report image end")

    def save(self, h5_path=None, tflite_path=None):
        if h5_path:
            self.log.i("save model as .h5 file")
            if not h5_path.endswith(".h5"):
                if os.path.isdir(h5_path):
                    h5_path = os.path.join(h5_path, "classifier.h5")
                else:
                    h5_path += ".h5"
            if not self.model:
                raise Exception("no model defined")
            self.model.save(h5_path)
        if tflite_path:
            self.log.i("save model as .tflite file")
            if not tflite_path.endswith(".tflite"):
                if os.path.isdir(tflite_path):
                    tflite_path = os.path.join(tflite_path, "classifier.tflite")
                else:
                    tflite_path += ".tflite"
            import tensorflow as tf
            converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
            tflite_model = converter.convert()
            with open (tflite_path, "wb") as f:
                f.write(tflite_model)

    def infer(self, input):
        pass

    def get_sample_images(self, sample_num, copy_to_dir):
        if not self.datasets_dir or not os.path.exists(self.datasets_dir):
            raise Exception("datasets dir not exists")
        num_gen = self._get_sample_num(len(self.labels), sample_num)
        for label in self.labels:
            num = num_gen.__next__()
            images = os.listdir(os.path.join(self.datasets_dir, label))
            images = random.sample(images, num)
            for image in images:
                shutil.copyfile(os.path.join(self.datasets_dir, label, image), os.path.join(copy_to_dir, image))


    def _get_confusion_matrix(self, ):
        batch_size = 5
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        from tensorflow.keras.applications.mobilenet import preprocess_input
        valid_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
        valid_data = valid_gen.flow_from_directory(self.datasets_dir,
                target_size=[self.input_shape[0], self.input_shape[1]],
                color_mode='rgb',
                batch_size=batch_size,
                class_mode='sparse',
                shuffle=False
            )
        prediction    = self.model.predict_generator(valid_data, steps=valid_data.samples//batch_size, verbose=1)
        predict_labels = np.argmax(prediction, axis=1)
        true_labels = valid_data.classes
        if len(predict_labels) != len(true_labels):
            true_labels = true_labels[0:len(predict_labels)]
        cm = confusion_matrix(true_labels, predict_labels)
        return cm, valid_data.class_indices
        

    def _unpack_datasets(self, datasets_zip, datasets_dir=None, rm_dataset=True):
        '''
            uppack zip datasets to /temp, make /temp as tmpfs is recommend
            zip should be: 
                            datasets
                                   |
                                    ---- class1
                                   |
                                    ---- class2
            or: 
                        ---- class1
                        ---- class2
        '''
        if not datasets_dir:
            datasets_dir = os.path.join(tempfile.gettempdir(), "classifer_datasets")
            if rm_dataset:
                self.datasets_rm_dir = datasets_dir
                self.need_rm_datasets = True
        if not os.path.exists(datasets_dir):
            os.makedirs(datasets_dir)
        zip_file = zipfile.ZipFile(datasets_zip)
        for names in zip_file.namelist():
            zip_file.extract(names, datasets_dir)
        zip_file.close()
        dirs = []
        for d in os.listdir(datasets_dir):
            if d.startswith(".") or not os.path.isdir(os.path.join(datasets_dir, d)):
                continue
            dirs.append(d)
        if len(dirs) == 1: # sub dir
            root_dir = dirs[0]
            datasets_dir = os.path.join(datasets_dir, root_dir)
        elif len(dirs) > 1:
            pass
        else: # empty zip
            return None
        return datasets_dir

    def _get_labels(self, datasets_dir):
        labels = []
        for d in os.listdir(datasets_dir):
            if d.startswith(".") or d == "__pycache__":
                continue
            if os.path.isdir(os.path.join(datasets_dir, d)):
                labels.append(d)
        return labels

    def _is_label_data_valid(self, labels, max_classes_num = 15,  min_images_num=40, max_images_num=2000):
        '''
            labels len should >= 2
            and should be ascii letters, no Chinese or special words
            images number in every label should > 40
        '''
        if len(labels) <= 1:
            err_msg = "datasets no enough class or directory error"
            return False, err_msg
        if len(labels) > max_classes_num:
            err_msg = "datasets too much class or directory error, limit:{} classses".format(max_classes_num)
            return False, err_msg
        print(labels,"---------")
        for label in labels:
            if not isascii(label):
                return False, "class name(label) should not contain special letters"
            # check image number
            files = os.listdir(os.path.join(self.datasets_dir, label))
            if len(files) < min_images_num:
                return False, "no enough train images in one class, should > {}".format(min_images_num)
            if len(files) > max_images_num:
                return False, "too many train images in one class, should < {}".format(max_images_num)
        return True, ""
    
    def _is_datasets_shape_valid(self, datasets_dir, shape):
        from PIL import Image
        ok = True
        msg = ""
        num_gen = self._get_sample_num(len(self.labels), len(self.labels))
        for label in self.labels:
            num = num_gen.__next__()
            images = os.listdir(os.path.join(self.datasets_dir, label))
            images = random.sample(images, num)
            for image in images:
                path = os.path.join(self.datasets_dir, label, image)
                img = np.array(Image.open(path))
                if img.shape != shape:
                    msg += f"image {label}/{image} shape is {img.shape}, but require {shape}\n"
                    ok = False
        return ok, msg

    def on_warning_message(self, msg):
        self.log.w(msg)
        self.warning_msg.append(msg)


def train_on_progress(progress, msg):
    print("\n==============")
    print("progress:{}%, msg:{}".format(progress, msg))
    print("==============")

def test_main(datasets_zip, model_path, report_path, use_cpu=False):
    if not os.path.exists("out"):
        os.makedirs("out")
    log = Logger(file_path="out/train.log")
    try:
        gpu = gpu_utils.select_gpu(memory_require = 1*1024*1024*1024, tf_gpu_mem_growth=False)
    except Exception:
        gpu = None
    if gpu is None:
        if not use_cpu:
            log.e("no free GPU")
            return 1
        log.i("no GPU, will use [CPU]")
    else:
        log.i("select", gpu)
    classifier = Classifier(datasets_zip=datasets_zip, logger=log)
    classifier.train(epochs=2, progress_cb=train_on_progress)
    classifier.report(report_path)
    classifier.save(model_path)

def test():
    if len(sys.argv) >= 4:
        test_main(sys.argv[1], sys.argv[2], sys.argv[3], use_cpu=True)
    else:
        test_main("./../../../../design/assets/test_classifier_datasets.zip",
                "out/classifier.h5",
                "out/report.jpg", use_cpu=True)

if __name__ == "__main__":
    '''
        arg: datasets_zip_file out_h5_model_path out_report_image_path
    '''
    test()

