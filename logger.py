import tensorboardX
import logging



class Logger():
    def __init__(self, experiment_id):

        format = "%(asctime)s %(threadName)s %(levelname)s: %(message)s"
        log_formatter = logging.Formatter(format)
        logging.basicConfig(level=logging.INFO, format=format)
        root_logger = logging.getLogger()

        # add file handle
        filename = os.path.join(dir, datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "pid-%d" %(os.getpid()) + ".log")
        file_handler = logging.FileHandler(filename)
        file_handler.setFormatter(log_formatter)
        root_logger.addHandler(file_handler)


    def flush(self):
        pass
    
    def add_epoch_info(self, epoch_num, total_frames, total_time, infos):
        pass