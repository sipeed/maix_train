
import os
from datetime import datetime, timedelta


class Train_Base:
    def __init__(self, classes, anchors, batch_size, input_size, input_layout, temp_dir, log, device="cuda"):
        self.save_path = os.path.join(temp_dir, "weights")
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.train_time = None
        self.train_last_iter = 0
        self.batch_size = batch_size

    def estimated_remain_time(self, epoch, max_epoch, iter_i, iter_total, parallel = 1):
        '''
            @epoch [0, max_epoch)
        '''
        remain = timedelta(0)
        if parallel == 1:
            max_iter  = max_epoch * iter_total
            curr_iter = epoch * iter_total + iter_i
            if not self.train_time is None:
                iter_num = curr_iter - self.train_last_iter
                interval = datetime.now() - self.train_time
                remain = interval * ((max_iter - curr_iter) / iter_num)
            self.train_time = datetime.now()
            self.train_last_iter = curr_iter
        else:
            raise NotImplementedError()
        return remain 


