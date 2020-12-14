#!python3

import argparse, os, sys, shutil


def main():
    supported_types = ["classifier", "detector"]
    curr_dir = os.path.abspath(os.path.dirname(__file__))
    parser = argparse.ArgumentParser(description="train model", usage='''
        python3 main.py -z "datasets zip file" init
    then
        python3 main.py -z "datasets zip file" train
        or  python3 main.py -d "datasets directory" train
''')
    parser.add_argument("-t", "--type", type=str, help="train type, classifier or detector", choices=supported_types, default="classifier")
    parser.add_argument("-z", "--zip", type=str, help="datasets zip file path", default="")
    parser.add_argument("-d", "--datasets", type=str, help="datasets directory", default="")
    parser.add_argument("-c", "--config", type=str, help="config file", default=os.path.join(curr_dir, "instance", "config.py"))
    parser.add_argument("-o", "--out", type=str, help="out directory", default=os.path.join(curr_dir, "out"))
    parser.add_argument("cmd", help="command", choices=["train", "init"])
    args = parser.parse_args()
    # init
    dst_config_path = os.path.join(curr_dir, "instance", "config.py")
    if args.cmd == "init":
        instance_dir = os.path.join(curr_dir, "instance")
        if not os.path.exists(instance_dir):
            os.makedirs(instance_dir)
        copy_config = True
        if os.path.exists(dst_config_path):
            print("[WARNING] instance/config.py already exists, sure to rewrite it? [yes/no]")
            ensure = input()
            if ensure != "yes":
                copy_config = False
        if copy_config:
            shutil.copyfile(os.path.join(curr_dir, "train", "config_template.py"), dst_config_path)
        print("init done, please edit instance/config.py")
        return 0
    if not os.path.exists(dst_config_path):
        print("python3 train.py init first")
        return -1

    from train import Train, TrainType
    is_zip = os.path.exists(args.zip)
    if not is_zip  and not os.path.exists(args.datasets):
        print("[ERROR] arg -d or -z is essential")
        return -1
    if args.type == "classifier":
        train_task = Train(TrainType.CLASSIFIER,  args.zip, args.datasets, args.out)
    elif args.type == "detector":
        train_task = Train(TrainType.DETECTOR,  args.zip, args.datasets, args.out)
    else:
        print("[ERROR] train type not support only support: {}".format(", ".join(supported_types)))
    train_task.train()
    return 0


if __name__ == "__main__":
    sys.exit(main())


