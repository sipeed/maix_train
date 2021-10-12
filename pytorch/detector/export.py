from genericpath import exists
from operator import mul
import subprocess
import torch.onnx
import torch
import numpy as np
import os

def torch_to_onnx(net, input_shape, out_name="out/model.onnx", input_names=["input0"], output_names=["output0"], device="cpu"):
    batch_size = 1
    if len(input_shape) == 3:
        x = torch.randn(batch_size, input_shape[0], input_shape[1], input_shape[2], dtype=torch.float32).to(device)
    elif len(input_shape) == 1:
        x = torch.randn(batch_size, input_shape[0], dtype=torch.float32).to(device)
    else:
        raise Exception("not support input shape")
    print("input shape:", x.shape)
    # torch.onnx._export(net, x, "out/conv0.onnx", export_params=True)
    torch.onnx.export(net, x, out_name, export_params=True, input_names = input_names, output_names=output_names)

def onnx_to_ncnn(input_shape, onnx="out/model.onnx", ncnn_param="out/conv0.param", ncnn_bin = "out/conv0.bin"):
    # onnx2ncnn tool compiled from ncnn/tools/onnx, and in the buld dir
    cmd = f"onnx2ncnn {onnx} {ncnn_param} {ncnn_bin}"
    os.system(cmd)
    with open(ncnn_param) as f:
        content = f.read().split("\n")
        if len(input_shape) == 1:
            content[2] += " 0={}".format(input_shape[0])
        else:
            content[2] += " 0={} 1={} 2={}".format(input_shape[2], input_shape[1], input_shape[0])
        content = "\n".join(content)
    with open(ncnn_param, "w") as f:
        f.write(content)

def ncnn_to_awnn(input_size, ncnn_param, ncnn_bin, quantize_images_path, mean = (127.5, 127.5, 127.5), norm = (0.0078125, 0.0078125, 0.0078125),
                 threads = 8, temp_dir = "out/temp",
                 awnn_param = None,
                 awnn_bin = None,
                 awnn_tools_cmd = "awnntools"):
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    if not awnn_param:
        tmp = os.path.splitext(ncnn_param)
        awnn_param = f'{tmp[0]}_awnn.param'
    if not awnn_bin:
        tmp = os.path.splitext(ncnn_bin)
        awnn_bin = f'{tmp[0]}_awnn.bin'
    # optimize
    cmd1 = f'{awnn_tools_cmd} optimize {ncnn_param} {ncnn_bin} {temp_dir}/opt.param {temp_dir}/opt.bin'

    # calibrate
    cmd2 = f'{awnn_tools_cmd} calibrate -p="{temp_dir}/opt.param" -b="{temp_dir}/opt.bin" -i="{quantize_images_path}"  -m="{", ".join([str(m) for m in mean])}" -n="{", ".join([str(m) for m in norm])}" -o="{temp_dir}/opt.table" -s="{", ".join([str(m) for m in input_size])}" -c="swapRB" -t={threads}'

    # quantize
    cmd3 = f'{awnn_tools_cmd} quantize {temp_dir}/opt.param {temp_dir}/opt.bin  {awnn_param} {awnn_bin} {temp_dir}/opt.table'
    cmd = f'{cmd1} && {cmd2} && {cmd3}'
    print(f"please execute cmd mannually:\n{cmd}")

def gen_input(input_shape, input_img=None, out_img_name="out/img.jpg", out_bin_name="out/input_data.bin", norm_int8=False):
    from PIL import Image
    if not input_img:
        input_img = (255, 0, 0)
    if type(input_img) == tuple:
        img = Image.new("RGB", (input_shape[2], input_shape[1]), input_img)
    else:
        img = Image.open(input_img)
        img = img.resize((input_shape[2], input_shape[1]))
    img.save(out_img_name)
    with open(out_bin_name, "wb") as f:
        print("norm_int8:", norm_int8)
        if not norm_int8:
            f.write(img.tobytes())
        else:
            data = (np.array(list(img.tobytes()), dtype=np.float)-128).astype(np.int8)
            f.write(bytes(data))

def get_net(net_type, classes, input_size, saved_state_path, log, anchor_len = 5, device = "cpu"):
    root = os.path.abspath(os.path.dirname(__file__))
    detectors_path = os.path.join(root, "detectors")
    sys.path.insert(0, detectors_path)
    detector = __import__(net_type)
    tester = detector.Test(
                        classes, [[1, 1] for i in range(anchor_len)],
                        input_size,
                        saved_state_path,
                        log,
                        device = device
                    )
    tester.net.post_process = False
    return tester.net

def save_images(dataset, awnn_quantize_images_path, input_size, max_num = -1):
    from progress.bar import Bar
    import cv2
    import random

    bar = Bar('save images', max=len(dataset))
    index = [i for i in range(len(dataset))]
    random.shuffle(index)
    index = index[:min(len(dataset), max_num)]
    for i, idx in enumerate(index):
        img = dataset.pull_image(idx, get_img_path=False)
        img = cv2.resize(img, input_size)
        save_path = os.path.join(awnn_quantize_images_path, f"{i}.jpg")
        cv2.imwrite(save_path, img)
        bar.next()
    bar.finish()

if __name__ == "__main__":
    import sys

    def main():
        from test import Net_Test
        from logger import Logger
        import os
        from dataset import Dataset_Folder, Dataset_VOC
        from augmentations import SSDAugmentationTest
        import shutil
        import cv2
        import multiprocessing

        #### config ####
        dataset_name = "cards2"
        classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "mouse", "microbit", "ruler", "cat", "peer", "ship", "apple", "car", "pan", "dog", "umbrella", "airplane", "clock", "grape", "cup", "left", "right", "front", "stop", "back"]
        param_path = "out/yolov2_slim/weights/epoch_460.pth"

        dataset_name = "lobster_5classes"
        classes = ["right", "left", "back", "front", "others"]
        param_path = "out/weights_save/lobster_5class_epoch_460.pth"

        is_val_dataset = True
        input_shape = (3, 224, 224)

        dataset_path = f"datasets/{dataset_name}"
        generate_images_from_val_images = True
        awnn_quantize_images_path = f"out/quantize_images/{dataset_name}"
        device = "cpu"
        max_sample_image_num = 500
        #### config end ####

        export_dir = f"out/export/{dataset_name}" 
        onnx_path = os.path.join(export_dir, f"{dataset_name}.onnx")
        ncnn_param_path = os.path.join(export_dir, f"{dataset_name}.param")
        ncnn_bin_path   = os.path.join(export_dir, f"{dataset_name}.bin")
        input_size = input_shape[1:][::-1]

        log = Logger()
        if not os.path.exists(export_dir):
            os.makedirs(export_dir)

        log.i("get net")
        net = get_net("yolov2_slim", classes, input_size, param_path, log, device=device)

        # onnx
        log.i("export onnx model")
        torch_to_onnx(net,
                    input_shape,
                    out_name=onnx_path,
                    device=device)
        log.i("export onnx ok")

        # ncnn
        log.i("export ncnn model ( onnx_to_ncnn cmd tool is needed )")
        onnx_to_ncnn(input_shape, onnx = onnx_path, ncnn_param = ncnn_param_path, ncnn_bin = ncnn_bin_path)
        log.i("export ncnn ok")

        # awnn
        log.i("prepare images for awnn")
        if generate_images_from_val_images:
            log.i("generate images for awnn quantize")
            if is_val_dataset:
                dataset = Dataset_VOC(classes, dataset_path, sets=["val"], log = log)
            else:
                dataset = Dataset_Folder(dataset_path,
                        transform = SSDAugmentationTest(size=input_size, mean=(0.5, 0.5, 0.5), std=(128/255.0, 128/255.0, 128/255.0)),
                        log=log
                        )
            if not os.path.exists(awnn_quantize_images_path):
                os.makedirs(awnn_quantize_images_path)
            elif len(os.listdir(awnn_quantize_images_path)) > 0:
                log.w(f"path {awnn_quantize_images_path} already exists, clear first? (y: yes, n: no)")
                while 1:
                    r = input("input to continue:\n\ty: clear dir, n: not clear dir:\n")
                    if r.lower() == "y" or r.lower() == "yes":
                        log.i("clear old quantize images")
                        shutil.rmtree(awnn_quantize_images_path)
                        os.makedirs(awnn_quantize_images_path)
                        log.i("save images, datasets len: ", len(dataset))
                        if len(dataset) == 0:
                            log.w("please check images dir ", dataset_path)
                        save_images(dataset, awnn_quantize_images_path, input_size, max_num = max_sample_image_num)
                        log.i("save images end")
                        break
                    elif r.lower() == "n" or r.lower() == "no":
                        log.i("images not update")
                        break
            else:
                log.i("save images, datasets len: ", len(dataset))
                if len(dataset) == 0:
                    log.w("please check images dir ", dataset_path)
                save_images(dataset, awnn_quantize_images_path, input_size, max_num = max_sample_image_num)
                log.i("save images end")

        if len(os.listdir(awnn_quantize_images_path)) <= 0 or not os.path.exists(awnn_quantize_images_path):
            log.e(f"awnn_quantize_images_path {awnn_quantize_images_path} not valid( not exists or empty )")
            return 1
        log.i("prepare images for awnn end, quantize images path: {awnn_quantize_images_path}")
        log.i(f"generate awnn model")
        ncnn_to_awnn(input_size, ncnn_param_path, ncnn_bin_path, awnn_quantize_images_path, threads=multiprocessing.cpu_count())
        log.i("export awnn model complete")
        return 0
    sys.exit(main())
