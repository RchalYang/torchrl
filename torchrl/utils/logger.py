import tensorboardX
import logging
import shutil
import os
import numpy as np
from tabulate import tabulate
import sys
import json
import csv

class Logger():
    def __init__(self, experiment_id, env_name, seed, params, log_dir = "./log"):
        """
        Initialize workdir.

        Args:
            self: (todo): write your description
            experiment_id: (int): write your description
            env_name: (str): write your description
            seed: (int): write your description
            params: (dict): write your description
            log_dir: (str): write your description
        """

        self.logger = logging.getLogger("{}_{}_{}".format(experiment_id,env_name,str(seed)))

        self.logger.handlers = []
        self.logger.propagate = False
        sh = logging.StreamHandler(sys.stdout)
        format = "%(asctime)s %(threadName)s %(levelname)s: %(message)s"
        formatter = logging.Formatter(format)
        sh.setFormatter(formatter)
        sh.setLevel(logging.INFO)
        self.logger.addHandler( sh )
        self.logger.setLevel(logging.INFO)

        work_dir = os.path.join( log_dir, experiment_id, env_name, str(seed) )
        self.work_dir = work_dir
        if os.path.exists( work_dir ):
            shutil.rmtree(work_dir)
        self.tf_writer = tensorboardX.SummaryWriter(work_dir)

        self.csv_file_path = os.path.join(work_dir, 'log.csv')
        # self.csv_file = open(os.path.join(work_dir, 'log.csv'), "a")
        # self.csv_writer = csv.writer(self.csv_file)

        self.update_count = 0
        self.stored_infos = {}

        with open( os.path.join(work_dir, 'params.json'), 'w' ) as output_param:
            json.dump( params, output_param, indent = 2 )

        self.logger.info("Experiment Name:{}".format(experiment_id))
        self.logger.info(
            json.dumps(params, indent = 2 )
        )

    def log(self, info):
        """
        Log a message

        Args:
            self: (todo): write your description
            info: (todo): write your description
        """
        self.logger.info(info)

    def add_update_info(self, infos):
        """
        Add info about infos. infos.

        Args:
            self: (todo): write your description
            infos: (todo): write your description
        """

        for info in infos:
            if info not in self.stored_infos :
                self.stored_infos[info] = []
            self.stored_infos[info].append( infos[info] )

        self.update_count += 1

    def add_epoch_info(self, epoch_num, total_frames, total_time, infos, csv_write=True):
        """
        Add epoch info to csv file.

        Args:
            self: (todo): write your description
            epoch_num: (int): write your description
            total_frames: (todo): write your description
            total_time: (todo): write your description
            infos: (todo): write your description
            csv_write: (bool): write your description
        """
        if csv_write:
            if epoch_num == 0:
                csv_titles = ["EPOCH", "Time Consumed", "Total Frames"]
            csv_values = [epoch_num, total_time, total_frames]

        self.logger.info("EPOCH:{}".format(epoch_num))
        self.logger.info("Time Consumed:{}s".format(total_time))
        self.logger.info("Total Frames:{}s".format(total_frames))

        tabulate_list = [["Name", "Value"]]

        for info in infos:
            self.tf_writer.add_scalar(info, infos[info], total_frames)
            tabulate_list.append([info, "{:.5f}".format( infos[info]) ])
            if csv_write:
                if epoch_num == 0:
                    csv_titles += [info]
                csv_values += ["{:.5f}".format(infos[info])]

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
                if csv_write:
                    if epoch_num == 0:
                        csv_titles += ["{}_{}".format(info, name)]
                    csv_values += ["{:.5f}".format(processed_info)]

            tabulate_list.append( temp_list )
        #clear
        self.stored_infos = {}
        if csv_write:
            with open(self.csv_file_path, 'a') as f:
                self.csv_writer = csv.writer(f)
                if epoch_num == 0:
                    self.csv_writer.writerow(csv_titles)
                self.csv_writer.writerow(csv_values)

        print( tabulate(tabulate_list) )
