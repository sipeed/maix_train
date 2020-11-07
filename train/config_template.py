


import os
curr_dir = os.path.abspath(os.path.dirname(__file__))


# kmodel convert
                 # "/ncc/ncc"  # download from https://github.com/kendryte/nncase/releases/tag/v0.1.0-rc5
ncc_kmodel_v3 =  os.path.join(curr_dir, "..", "tools", "ncc", "ncc_v0.1/ncc")  
sample_image_num = 20       # convert kmodel sample image (for quantizing)

# train
allow_cpu = True # True

# classifier
classifier_train_gpu_mem_require = 2*1024*1024*1024
classifier_train_epochs = 40
classifier_train_batch_size = 5
classifier_train_max_classes_num = 15
classifier_train_one_class_min_img_num = 40            # 一个类别中至少需要的样本数量
classifier_train_one_class_max_img_num = 2000          # 一个类别中最多需要的样本数量
classifier_result_file_name_prefix = "maixhub_classifier_result"

# detector
detector_train_gpu_mem_require = 2*1024*1024*1024
detector_train_epochs = 40
detector_train_batch_size = 5
detector_train_learn_rate = 1e-4
detector_train_max_classes_num = 15         # 最多能训练多少类
detector_train_one_class_min_img_num = 100            # 一个类别中至少需要的样本数量
detector_train_one_class_max_img_num = 2000           # 一个类别中最多需要的样本数量
detector_result_file_name_prefix = "maixhub_detector_result"



