
This is a temporary folder for train classifier with pytorch


## Usage

### classifier

* Prepare dataset, each class has one folder with class name
* Edit classifier_resnet_train.py's config section
* `python classifier_resnet_train.py`

You can test your model after train complete by `python classifier_resnet_test.py images_folder_path`


### detector

* Prepare dataset at `detectore/datasets`dir, format is the same as VOC data, e.g.

```
datasets
└── lobster_5classes
    ├── Annotations
            └── *.xml
    ├── ImageSets
            └── Main
                    ├── train.txt
                    └── val.txt
    └── JPEGImages
            └── *.jpg
```

* Edit main part of `train.py`, set `classes` `dataset_name` etc.

* Execute `python train.py` to start train

* find the best mAP epoch, use the weight file in `out` dir to test model
Edit `test.py` first and execute
```
python test.py
```

> Some code from https://github.com/yjh0410/yolov2-yolov3_PyTorch

