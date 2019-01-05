import tensorboardX
import logging
import shutil
import os
import numpy as np
from tabulate import tabulate

class Logger():
    def __init__(self, experiment_id, env_name, seed):

        format = "%(asctime)s %(threadName)s %(levelname)s: %(message)s"
        log_formatter = logging.Formatter(format)
        logging.basicConfig(level=logging.INFO, format=format)
        root_logger = logging.getLogger()

        # # add file handle
        # filename = os.path.join(dir, datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "pid-%d" %(os.getpid()) + ".log")
        # file_handler = logging.FileHandler(filename)
        # file_handler.setFormatter(log_formatter)
        # root_logger.addHandler(file_handler)
        work_dir = os.path.join("log", env_name, str(seed),  experiment_id )
        if os.path.exists( work_dir ):
            shutil.rmtree(work_dir)
        self.tf_writer = tensorboardX.SummaryWriter(work_dir)
        self.update_count = 0
        self.stored_infos = {}

    def log(self, info):
        logging.info(info)

    def add_update_info(self, infos):
        for info in infos:
            self.tf_writer.add_scalar( info, infos[info], self.update_count )

            if info not in self.stored_infos :
                self.stored_infos[info] = []

            self.stored_infos[info].append( infos[info] )
            
        self.update_count += 1
    
    def add_epoch_info(self, epoch_num, total_frames, total_time, infos):
        logging.info("EPOCH:{}".format(epoch_num))
        logging.info("Time Consumed:{}s".format(total_time))

        tabulate_list = []
        for info in infos:
            self.tf_writer.add_scalar( info, infos[info], total_frames )
            tabulate_list.append([ info, infos[info] ])

        tabulate_list.append(["Name", "Mean", "Min", "Max" ])
        
        for info in self.stored_infos:
            temp_list = [info]

            temp_list.append( np.mean(self.stored_infos[info]) )
            temp_list.append( np.std(self.stored_infos[info]) )
            temp_list.append( np.max(self.stored_infos[info]) )
            temp_list.append( np.min(self.stored_infos[info]) )
            
            tabulate_list.append( temp_list )
        print( tabulate(tabulate_list) )
