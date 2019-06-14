import os


def read_dir(path):
    ir_list = os.listdir(path)
    return [os.path.join(path, ir) for ir in ir_list]
