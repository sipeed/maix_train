'''
    train main class

    @author neucrack@sipeed
    @license Apache 2.0 © 2020 Sipeed Ltd
'''

import os, sys
root_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.append(root_path)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


from classifier import Classifier
from detector import Detector
import requests
import tempfile
import shutil
from utils import gpu_utils, isascii
from utils.logger import Logger, Fake_Logger
from instance import config
import time
from datetime import datetime
import subprocess
import zipfile
import traceback
import json
from enum import Enum

class TrainType(Enum):
    CLASSIFIER = 0
    DETECTOR   = 1

class TrainFailReason(Enum):
    ERROR_NONE     = 0
    ERROR_INTERNAL = 1
    ERROR_DOWNLOAD_DATASETS = 2
    ERROR_NODE_BUSY = 3
    ERROR_PARAM     = 4
    ERROR_CANCEL    = 5


class Train():
    def __init__(self, train_type: TrainType,
                 datasets_zip,
                 dataset_dir,
                 out_dir):
        '''
            creat /temp/train_temp dir to train
        '''
        self.train_type = train_type
        self.datasets_zip_path = datasets_zip
        self.dataset_dir = dataset_dir
        self.temp_dir = out_dir
        assert os.path.exists(datasets_zip) or os.path.exists(dataset_dir)
        if os.path.exists(dataset_dir):
            self.datasets_dir = dataset_dir
        else:
            self.datasets_dir = ""
        self.temp_datasets_dir = os.path.join(self.temp_dir, "datasets")
        self.result_dir = os.path.join(self.temp_dir, "result")
        self.clean_temp_files()
        os.makedirs(self.temp_dir)
        if os.path.exists(self.result_dir):
            shutil.rmtree(self.result_dir)
        os.makedirs(self.result_dir)
        self.dataset_sample_images_path = os.path.join(self.temp_dir, "sample_images")
        os.makedirs(self.dataset_sample_images_path)
        self.log_file_path = os.path.join(self.temp_dir, "train_log.log")
        self.result_report_img_path = os.path.join(self.result_dir, "report.jpg")
        self.result_kmodel_path = os.path.join(self.result_dir, "m.kmodel")
        self.result_labels_path  = os.path.join(self.result_dir, "labels.txt")
        self.result_boot_py_path = os.path.join(self.result_dir, "boot.py")
        self.tflite_path = os.path.join(self.temp_dir, "m.tflite")
        self.final_h5_model_path = os.path.join(self.temp_dir, "m.h5")
        self.best_h5_model_path  = os.path.join(self.temp_dir, "m_best.h5")

        self.log = Logger(file_path=self.log_file_path)
    
    def __del__(self):
        # self.clean_temp_files()
        pass
    
    def clean_temp_files(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)


    def __on_progress(self, percent, msg):  # flag: progress
        self.log.i(f"progress: {percent}%, {msg}")
    
    def __on_success(self, result_url, warn):
        self.log.i(f"success: out_dir: {result_url}")
        if warn:
            self.log.w(f"warnings:\n {warn}")

    def __on_fail(self, reson, msg, warn):
        self.log.e(f"failed: {reson}, {msg}")
        if warn:
            self.log.w(f"warnings:\n {warn}")

    def __on_train_progress(self, percent, msg):  # flag: progress
        percent = percent*0.97 + 1
        self.log.i(f"progress: {percent}%, {msg}")

    def train(self):
        warning_msg = ""
        try:
            result_url, warning_msg = self.train_process(self.log)
            self.__on_success(result_url, warning_msg)
        except Exception as e:
            info = e.args[0]
            if type(info) == tuple and len(info) == 2:
                reason = info[0]
                msg = info[1]
                self.__on_fail(reason, msg, warning_msg)
            else:
                self.__on_fail(TrainFailReason.ERROR_INTERNAL, "node error:{}".format(e), warning_msg)

    def train_process(self, log):
        '''
            raise Exception if error occurred, a tuple: (TrainFailReason, error_message)
            @return result url
        '''
        self.__on_progress(0, "start") # flag: progress
        self.__on_progress(1, "start train") # flag: progress
        
        if self.train_type == TrainType.CLASSIFIER:
            obj, prefix = self.classifier_train(log = log)
        elif self.train_type == TrainType.DETECTOR:
            obj, prefix = self.detector_train(log = log)
        else:
            raise Exception(( "error train type, not suport"))
        
        # check warnings
        result_warning_msg = ""
        result_warning_msg_path = os.path.join(self.result_dir, "warning.txt")
        if len(obj.warning_msg) > 0:
            result_warning_msg += "=========================================================================\n"
            result_warning_msg += "train warnings: these warn info may lead train error(accuracy loss), please check carefully\n"
            result_warning_msg += "=========================================================================\n"
            result_warning_msg += "训练警告： 这些警告信息可能导致训练误差，请务必仔细检查\n"
            result_warning_msg += "=========================================================================\n\n\n"
            for msg in obj.warning_msg:
                result_warning_msg += "{}\n\n".format(msg)
            with open(result_warning_msg_path, "w") as f:
                f.write(result_warning_msg)

        # pack zip
        log.i("pack result to zip file")
        time_now = datetime.now().strftime("%Y_%m_%d__%H_%M")
        result_dir_name = "{}_{}".format(prefix, time_now)
        result_zip_name = "{}.zip".format(result_dir_name)
        result_dir = os.path.join(os.path.dirname(self.result_dir), result_dir_name)
        os.rename(self.result_dir, result_dir)
        root_dir = os.path.join(self.temp_dir, "result_root_dir")
        os.mkdir(root_dir)
        shutil.move(result_dir, root_dir) # 移动 result 文件夹, 到一个 root_dir下,用以压缩
        result_zip = os.path.join(self.temp_dir, result_zip_name)
        try: 
            # self.zip_dir(root_dir, result_zip)
            # for old maixhub compatibility
            self.zip_dir(os.path.join(root_dir, result_dir_name), result_zip)
        except Exception:
            log.e("zip result fail")
            raise Exception((TrainFailReason.ERROR_INTERNAL, "zip result error"))

        # progress 99%
        self.__on_progress(99, "pack ok") # flag: progress

        # complete
        self.__on_progress(100, "task complete") # flag: progress
        log.i("OK, task complete, result uri: {}".format(result_zip))
        return result_zip, result_warning_msg

    def classifier_train(self, log):
        # 检测 GPU 可用,选择一个可用的 GPU 使用
        try:
            gpu = gpu_utils.select_gpu(memory_require = config.classifier_train_gpu_mem_require, tf_gpu_mem_growth=False)
        except Exception:
            gpu = None
        if gpu is None:
            if not config.allow_cpu:
                log.e("no free GPU")
                raise Exception((TrainFailReason.ERROR_NODE_BUSY, "node no enough GPU or GPU memory and not support CPU train"))
            log.i("no GPU, will use [CPU]")
        else:
            log.i("select", gpu)

        # 启动训练
        try:
            classifier = Classifier(datasets_zip=self.datasets_zip_path, datasets_dir=self.datasets_dir, unpack_dir = self.temp_datasets_dir,
                                    logger=log,
                                    max_classes_num=config.classifier_train_max_classes_num,
                                    min_images_num=config.classifier_train_one_class_min_img_num,
                                    max_images_num=config.classifier_train_one_class_max_img_num,
                                    allow_reshape=False)
        except Exception as e:
            log.e("train datasets not valid: {}".format(e))
            raise Exception((TrainFailReason.ERROR_PARAM, "datasets not valid: {}".format(str(e))))
        try:
            classifier.train(epochs=config.classifier_train_epochs, batch_size=config.classifier_train_batch_size, progress_cb=self.__on_train_progress)
        except Exception as e:
            log.e("train error: {}".format(e))
            traceback.print_exc()
            raise Exception((TrainFailReason.ERROR_INTERNAL, "error occurred when train, error: {}".format(str(e)) ))

        # 训练结束, 生成报告
        log.i("train ok, now generate report")
        classifier.report(self.result_report_img_path)

        # 生成 kmodel
        log.i("now generate kmodel")
        classifier.save(self.tflite_path+".h5", tflite_path=self.tflite_path)
        classifier.get_sample_images(config.sample_image_num, self.dataset_sample_images_path)
        ok, msg = self.convert_to_kmodel(self.tflite_path, self.result_kmodel_path, config.ncc_kmodel_v3, self.dataset_sample_images_path)
        if not ok:
            log.e("convert to kmodel fail")
            raise Exception((TrainFailReason.ERROR_INTERNAL, "convert kmodel fail: {}".format(msg) ))

        # 拷贝模板文件
        log.i("copy template files")
        template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "classifier", "template")
        self.__copy_template_files(template_dir, self.result_dir)

        # 写入 label 文件
        replace = 'labels = ["{}"]'.format('", "'.join(classifier.labels))
        with open(self.result_labels_path, "w") as f:
            f.write(replace)
        with open(self.result_boot_py_path) as f:
            boot_py = f.read()
        with open(self.result_boot_py_path, "w") as f:
            target = 'labels = [] # labels'
            boot_py = boot_py.replace(target, replace)
            target = 'sensor.set_windowing((224, 224))'
            replace = 'sensor.set_windowing(({}, {}))'.format(classifier.input_shape[1], classifier.input_shape[0])
            boot_py = boot_py.replace(target, replace)
            f.write(boot_py)

        return classifier, config.classifier_result_file_name_prefix

    def detector_train(self, log):
                # 检测 GPU 可用,选择一个可用的 GPU 使用
        try:
            gpu = gpu_utils.select_gpu(memory_require = config.detector_train_gpu_mem_require, tf_gpu_mem_growth=False)
        except Exception:
            gpu = None
        if gpu is None:
            if not config.allow_cpu:
                log.e("no free GPU")
                raise Exception((TrainFailReason.ERROR_NODE_BUSY, "node no enough GPU or GPU memory and not support CPU train"))
            log.i("no GPU, will use [CPU]")
        else:
            log.i("select", gpu)

        # 启动训练
        try:
            detector = Detector(input_shape=(224, 224, 3),
                                datasets_zip=self.datasets_zip_path,
                                datasets_dir=self.datasets_dir,
                                unpack_dir = self.temp_datasets_dir,
                                logger=log,
                                max_classes_limit = config.detector_train_max_classes_num,
                                one_class_min_images_num=config.detector_train_one_class_min_img_num,
                                one_class_max_images_num=config.detector_train_one_class_max_img_num,
                                allow_reshape = False)
        except Exception as e:
            log.e("train datasets not valid: {}".format(e))
            raise Exception((TrainFailReason.ERROR_PARAM, "datasets not valid: {}".format(str(e))))
        try:

            detector.train(epochs=config.detector_train_epochs,
                    progress_cb=self.__on_train_progress,
                    save_best_weights_path = self.best_h5_model_path,
                    save_final_weights_path = self.final_h5_model_path,
                    jitter=False,
                    is_only_detect = False,
                    batch_size = config.detector_train_batch_size,
                    train_times = 5,
                    valid_times = 2,
                    learning_rate=config.detector_train_learn_rate,
                )
        except Exception as e:
            log.e("train error: {}".format(e))
            traceback.print_exc()
            raise Exception((TrainFailReason.ERROR_INTERNAL, "error occurred when train, error: {}".format(str(e)) ))

        # 训练结束, 生成报告
        log.i("train ok, now generate report")
        detector.report(self.result_report_img_path)

        # 生成 kmodel
        log.i("now generate kmodel")
        detector.save(tflite_path=self.tflite_path)
        detector.get_sample_images(config.sample_image_num, self.dataset_sample_images_path)
        ok, msg = self.convert_to_kmodel(self.tflite_path, self.result_kmodel_path, config.ncc_kmodel_v3, self.dataset_sample_images_path)
        if not ok:
            log.e("convert to kmodel fail")
            raise Exception((TrainFailReason.ERROR_INTERNAL, "convert kmodel fail: {}".format(msg) ))

        # 拷贝模板文件
        log.i("copy template files")
        template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "detector", "template")
        self.__copy_template_files(template_dir, self.result_dir)

        # 写入 label 文件
        replace = 'labels = ["{}"]'.format('", "'.join(detector.labels))
        with open(self.result_labels_path, "w") as f:
            f.write(replace)
        with open(self.result_boot_py_path) as f:
            boot_py = f.read()
        with open(self.result_boot_py_path, "w") as f:
            target = 'labels = [] # labels'
            boot_py = boot_py.replace(target, replace)
            target = 'anchors = [] # anchors'
            replace = 'anchors = [{}]'.format(', '.join(str(i) for i in detector.anchors))
            boot_py = boot_py.replace(target, replace)
            target = 'sensor.set_windowing((224, 224))'
            replace = 'sensor.set_windowing(({}, {}))'.format(detector.input_shape[1], detector.input_shape[0])
            boot_py = boot_py.replace(target, replace)
            f.write(boot_py)

        return detector, config.detector_result_file_name_prefix

    def __copy_template_files(self, src_dir,  dst_dir):
        files = os.listdir(src_dir)
        for f in files:
            shutil.copyfile(os.path.join(src_dir, f), os.path.join(dst_dir, f))

    def zip_dir(self, dir_path, out_zip_file_path):
        '''
            将目录打包成zip, 注意传的目录是根目录,是不会被打包进压缩包的,如果需要文件夹,要在这个目录下建立一个子文件夹
            root_dir
                   |
                   -- data_dir
                            -- data1
                            -- data2
            zip: 
                name.zip
                    |
                    -- data_dir
                                -- data1
                                -- data2
        '''
        shutil.make_archive(os.path.splitext(out_zip_file_path)[0], "zip", dir_path)
        

    def convert_to_kmodel(self, tf_lite_path, kmodel_path, ncc_path, images_path):
        '''
            @ncc_path ncc 可执行程序路径
            @return (ok, msg) 是否出错 (bool, str)
        '''
        p =subprocess.Popen([ncc_path, "-i", "tflite", "-o", "k210model", "--dataset", images_path, tf_lite_path, kmodel_path], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            output, err = p.communicate( )
            res = p.returncode
        except Exception as e:
            print("[ERROR] ", e)
            return False, str(e)
        res = p.returncode
        if res == 0:
            return True, "ok"
        else:
            print("[ERROR] ", res, output, err)
        return False, f"output:\n{output}\nerror:\n{err}"

if __name__ == "__main__":
    # train_task = Train(TrainType.CLASSIFIER,  "../datasets/test_classifier_datasets.zip", "", "../out")
    train_task = Train(TrainType.DETECTOR,  "../datasets/test_detector_xml_format.zip", "", "../out")
    train_task.train()

