import tensorboardX
import logging
import shutil
import os
import numpy as np
from tabulate import tabulate

class Logger():
    def __init__(self, experiment_id, env_name, seed):

        format = "%(asctime)s %(threadName)s %(levelname)s: %(message)s"
        logging.basicConfig(level=logging.INFO, format=format)

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

        tabulate_list = [["Name", "Value"]]
        for info in infos:
            self.tf_writer.add_scalar( info, infos[info], total_frames )
            tabulate_list.append([ info, infos[info] ])

        tabulate_list.append(["Name", "Mean", "Min", "Max" ])
        
        for info in self.stored_infos:
            temp_list = [info]

            temp_list.append( "{:.5f}".format(np.mean(self.stored_infos[info])) )
            temp_list.append( "{:.5f}".format(np.std(self.stored_infos[info])) )
            temp_list.append( "{:.5f}".format(np.max(self.stored_infos[info])) )
            temp_list.append( "{:.5f}".format(np.min(self.stored_infos[info])) )
            
            tabulate_list.append( temp_list )
        print( tabulate(tabulate_list) )
