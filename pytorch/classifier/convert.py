import torch.onnx
import torch
import numpy as np

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
    print("export onnx ok")

def onnx_to_ncnn(input_shape, onnx="out/model.onnx", ncnn_param="out/conv0.param", ncnn_bin = "out/conv0.bin"):
    import os
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

