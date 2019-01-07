import tensorboardX
import logging
import shutil
import os
import numpy as np
from tabulate import tabulate

class Logger():
    def __init__(self, experiment_id, env_name, seed, log_dir = "./log"):

        format = "%(asctime)s %(threadName)s %(levelname)s: %(message)s"
        logging.basicConfig(level=logging.INFO, format=format)

        work_dir = os.path.join( log_dir, experiment_id, env_name, str(seed) )
        if os.path.exists( work_dir ):
            shutil.rmtree(work_dir)
        self.tf_writer = tensorboardX.SummaryWriter(work_dir)

        self.update_count = 0
        self.stored_infos = {}

    def log(self, info):
        logging.info(info)

    def add_update_info(self, infos):

        for info in infos:
            if info not in self.stored_infos :
                self.stored_infos[info] = []
            self.stored_infos[info].append( infos[info] )
            
        self.update_count += 1
    
    def add_epoch_info(self, epoch_num, total_frames, total_time, infos):
        logging.info("EPOCH:{}".format(epoch_num))
        logging.info("Time Consumed:{}s".format(total_time))
        logging.info("Total Frames:{}s".format(total_frames))

        tabulate_list = [["Name", "Value"]]
        for info in infos:
            self.tf_writer.add_scalar( info, infos[info], total_frames )
            tabulate_list.append([ info, "{:.5f}".format( infos[info] ) ])

        tabulate_list.append([])
        
        
        method_list = [ np.mean, np.std, np.max, np.min ]
        name_list = [ "Mean", "Std", "Max", "Min" ]
        tabulate_list.append( ["Name"] + name_list )

        for info in self.stored_infos:

            temp_list = [info]
            for name, method in zip( name_list, method_list ):
                processed_info = method(self.stored_infos[info])
                self.tf_writer.add_scalar( "{}_{}".format( info, name ),
                    processed_info, total_frames )
                temp_list.append( "{:.5f}".format( processed_info ) )

            tabulate_list.append( temp_list )
        #clear
        self.stored_infos = {}

        print( tabulate(tabulate_list) )
