
from ast import arg
import os
import os.path
import sys
current_directory = os.getcwd()
sys.path.insert(0, current_directory)
import numpy as np
import torch
from src.models.DRM_eval import drm_eval
from src.models.modeling import ClassificationHead, CLIPEncoder, ImageClassifier
from src.args import parse_arguments
import logging
import random


def main(args):
    os.makedirs("expt_logs/" + args.exp_name, exist_ok=True)
    logging_path = "expt_logs/" + args.exp_name + "/" + "_BS" + str(
        args.batch_size) + "_WD" + str(args.wd) + "_LR" + str(args.lr) + "_run" + str(args.run)
    os.makedirs(logging_path, exist_ok=True)
    log_filename = logging_path + "/log.log"
    logging.basicConfig(filename=log_filename,
                        format='%(asctime)s %(message)s',
                        filemode='w')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    clip_encoder = CLIPEncoder(args, keep_lang=True)
    logger.info(args)
    if args.checkpoint_path!=None:
        clip_encoder = clip_encoder.load(args.checkpoint_path)
    finetuned_checkpoint = drm_eval(args, clip_encoder,logger)


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
