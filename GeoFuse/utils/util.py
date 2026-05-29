import logging
import random
import numpy as np
import torch

def get_logger():
    logger = logging.getLogger("GeoFuse")
    if not logger.handlers:
        logging.basicConfig(level=logging.INFO)
    return logger

def init_logger(log_file):
    logger = logging.getLogger("GeoFuse")
    if logger.handlers:
        logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False 
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger

def set_random(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False