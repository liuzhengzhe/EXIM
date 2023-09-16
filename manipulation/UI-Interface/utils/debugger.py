import time
import datetime
import os
import sys
import numpy as np
import logging
import random
from shutil import copyfile
import torch


class MyDebugger():
    # pre_fix = r'D:\wavelet\Interaction_UI\debug'

    def __init__(self, model_name: str, select_chair=1, fix_rand_seed=None):
        if select_chair:
            self.pre_fix = 'debug'
        else:
            self.pre_fix = 'debug'

        if fix_rand_seed is not None:
            np.random.seed(seed=fix_rand_seed)
            random.seed(fix_rand_seed)
            torch.manual_seed(fix_rand_seed)
        if isinstance(model_name, str):
            self.model_name = model_name
        else:
            self.model_name = '_'.join(model_name)
        # self._debug_dir_name = os.path.join(os.path.dirname(__file__), MyDebugger.pre_fix,
        #                                     datetime.datetime.fromtimestamp(time.time()).strftime(
        #                                         f'%Y-%m-%d_%H-%M-%S_{self.model_name}'))

        self._debug_dir_name = os.path.join(os.path.dirname(__file__), self.pre_fix,
                                            datetime.datetime.fromtimestamp(time.time()).strftime(
                                                f'%Y-%m-%d_%H-%M-%S_{self.model_name}'))
        # self._debug_dir_name = os.path.join(os.path.dirname(__file__), self._debug_dir_name)
        print("=================== Program Start ====================")
        print(f"Output directory: {self._debug_dir_name}")
        self._init_debug_dir()


    def file_path(self, file_name):
        return os.path.join(self._debug_dir_name, file_name)

    def _init_debug_dir(self):
        # init root debug dir
        # if not os.path.exists(MyDebugger.pre_fix):
        #     os.mkdir(MyDebugger.pre_fix)
        if not os.path.exists(self.pre_fix):
            os.mkdir(self.pre_fix)
        os.mkdir(self._debug_dir_name)
        logging.info("Directory %s established" % self._debug_dir_name)


if __name__ == '__main__':
    debugger = MyDebugger('testing')
    # file can save in the path
    file_path = debugger.file_path('file_to_be_save.txt')
