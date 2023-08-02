import os
import sys

# from sklearn.utils import gen_even_slices
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))
import logging
from pathlib import Path
import shutil

def os_makedirs(folder_path):
    try:
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
    except FileExistsError:
        pass

def os_rmdirs(folder_path):
    try:
        dirPath = Path(folder_path)
        if dirPath.exists() and dirPath.is_dir():
            shutil.rmtree(dirPath)
    except FileExistsError:
        pass

def set_logger(log_path, log_name, level = 20, rewrite = True):
    '''Set the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `task_dir/train.log`.
    Example:
    logging.info('Starting training...')
    Args:
        log_path: (string) where to log
    '''
    
    logger = logging.Logger(log_name)
    if os.path.exists(log_path) and rewrite:
        os.remove(log_path) # os.remove can only delete a file with given file_path; os.rmdir() can delete a directory.
    log_file = Path(log_path)
    log_folder = log_file.parent
    os_makedirs(log_folder)
    log_file.touch(exist_ok=True)


    if level == 50:
        logger.setLevel(logging.CRITICAL)
    else:
        logger.setLevel(logging.INFO)

    fmt = logging.Formatter('[%(asctime)s] %(name)s: %(message)s', '%Y-%m-%d %H:%M:%S')

    class TqdmHandler(logging.StreamHandler):
        def __init__(self, formatter):
            logging.StreamHandler.__init__(self)
            self.setFormatter(formatter)

        def emit(self, record):
            msg = self.format(record)
            tqdm.write(msg)

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    logger.addHandler(TqdmHandler(fmt))

    return logger