train_scripts
===========

You can also train on [Maixhub.com](https://www.maixhub.com), 
just upload your datasets and you will get the result(kmodel and usage code)

## Train type

* Object classification(Mobilenet V1): judge class of image
* Object detection(YOLO v2): find a recognizable object in the picture


## Usage

### 0. Prepare

* only support `Linux`
* Prepare environment, use CPU or GPU to train
At your fist time train, CPU is recommended, just
```
pip3 install -r requirements.txt
```
or use aliyun's source if you are in China
```
pip3 install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
```

* [Download nncase](https://github.com/kendryte/nncase/releases/tag/v0.1.0-rc5) and unzip it to `tools/ncc/ncc_v0.1`, and the executable path is `tools/ncc/ncc_v0.1/ncc`
* `python3 train.py init`
* Edit `instance/config.py` according to your hardware
* Prepare dataset, in the `datasets` directory has some example datasets, input size if `224x224`
  or you just fllow [maixhub](https://www.maixhub.com/mtrain.html)'s conduct

### 1. Object classification (Mobilenet V1)

```
python3 train.py -t classifier -z datasets/test_classifier_datasets.zip train
```
or assign datasets directory
```
python3 train.py -t classifier -d datasets/test_classifier_datasets train
```

more command see`python3 train.py -h`


and you will see output in the `out` directory, packed as a zip file


### 2. Object detection (YOLO V2)


```
python3 train.py -t detector -z datasets/test_detector_xml_format.zip train
```

more command see`python3 train.py -h`

and you will see output in the `out` directory, packed as a zip file


## License

Apache 2.0, see [LICENSE](LICENSE)

