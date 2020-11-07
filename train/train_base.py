'''
    train base class

    @author neucrack@sipeed
    @license Apache 2.0 © 2020 Sipeed Ltd
'''


import abc

class Train_Base(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def __init__(self,
                    datasets_zip,
                    unpack_dir,
                    logger
                ):
        raise NotImplementedError        

    @abc.abstractmethod
    def train(self,
                epochs,
                progress_cb = None, # callback(percentage, msg)
                weights = None,
                batch_size = 5
             ):
        raise NotImplementedError

    @abc.abstractmethod
    def report(self, result_report_img_path):
        raise NotImplementedError

    @abc.abstractmethod
    def save(self, h5_path=None, tflite_path=None):
        raise NotImplementedError

    @abc.abstractmethod
    def get_sample_images(self, sample_num, copy_to_dir):
        '''
            copy images from datasets for quantization
        '''
        raise NotImplementedError

    def _get_sample_num(self, classes_num, sample_num):
        '''
            iter for get sample num in every class
        '''
        batch_size = sample_num // classes_num
        batch_size = 1 if batch_size==0 else batch_size
        for i in range(classes_num):
            if i * batch_size >= sample_num:
                return
            if i == classes_num - 1:
                yield sample_num - i * batch_size
            else:
                yield  batch_size
