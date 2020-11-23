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


## Use GPU

Use docker or install tensorflow with GPU in your local environment

Tensorflow's version should >= 2.0, tested on 2.1

### Use docker(recommend)

see [tensorflow official website](https://tensorflow.google.cn/install/docker) (或者可以参考[这篇教程](https://neucrack.com/p/116))

* Install docker
* Install [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)
* Pull docker image
```
docker pull neucrack/tensorflow-gpu-py3-jupyterlab
```
or
```
docker pull daocloud.io/neucrack/tensorflow-gpu-py3-jupyterlab
```
* Test environment
```
docker run --gpus all -it --rm neucrack/tensorflow-gpu-py3-jupyterlab python -c "import tensorflow as tf; print('-----version:{}, gpu:{}, 1+2={}'.format(tf.__version__, tf.test.is_gpu_available(), tf.add(1, 2).numpy()) );"
```
if output is`-----version:2.1.0, gpu:True, 1+2=3`, that's ok(maybe version can `> 2.1.0`)
* Create docker container
```shell
docker run --gpus all --name jupyterlab-gpu -it -p 8889:8889 -e USER_NAME=$USER -e USER_ID=`id -u $USER` -e GROUP_NAME=`id -gn $USER` -e GROUP_ID=`id -g $USER` -v /home/${USER}:/tf neucrack/tensorflow-gpu-py3-jupyterlab
```
If used daocloud, image name should be change to `daocloud.io/neucrack/tensorflow-gpu-py3-jupyterlab`

This will mount your`/home/$USER` directory to `/tf` directory of container, the `/tf` is the root dir of jupyterlab

Stop by `docker stop jupyterlab-gpu`, start again by `docker start jupyterlab-gpu`
To use `sudo` command, edit user password by
```shell
docker exec -it jupyterlab_gpu /bin/bash
passwd $USER
passwd root
```

* use jupyterlab

Open `http://127.0.0.1:8889/lab?` in browser, input token(see docker start log) and set new password

Use `docker stop jupyterlab-gpu` to stop server
Use `docker start jupyterlab-gpu` to start service again



### Install on local environment

refer to [tensorflow official website](https://tensorflow.google.cn/install/gpu)


## License

Apache 2.0, see [LICENSE](LICENSE)

