from asyncio.constants import LOG_THRESHOLD_FOR_CONNLOST_WRITES
import os
import copy
import time
import tqdm
import torch
import pandas as pd
import clip_m.clip as clip
from clip_m.loss import ClipLoss
from src.args import parse_arguments
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.models.eval import evaluate
from src.models.modeling import ClassificationHead, CLIPEncoder, ImageClassifier
from src.models.zeroshot import get_zeroshot_classifier
import src.datasets as datasets
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import pandas as pd
import src.templates as templates
import random
import gc
import json

def drm_eval(args, clip_encoder, logger):
    with open(args.cd_path) as json_file:
        cd_data = json.load(json_file)
    cd_data_key=list(cd_data.keys())
    template = getattr(templates, args.template)
    model = clip_encoder
    preprocess_fn = clip_encoder.train_preprocess
    image_enc = None
    clip_encoder.process_images = True
    devices = list(range(torch.cuda.device_count()))
    logger.info('Using devices' + str(devices))
    model = torch.nn.DataParallel(model, device_ids=devices)
    model.eval()



    classification_head_llm_new,classification_head_basic_new = get_zeroshot_classifier(
        args, model.module.model)
    classification_head_llm_new = classification_head_llm_new.cuda()
    classification_head_basic_new = classification_head_basic_new.cuda()
    epoch_stats={}
    stats=[]
    eval_results = evaluate(model, args, classification_head_llm_new,classification_head_basic_new,args.beta,
                        epoch_stats, logger)

    del classification_head_llm_new
    del classification_head_basic_new

    ood_acc = 0
    num_datasets = 0
    for k, v in epoch_stats.items():
        if 'Accuracy' in k:
            if k == 'ImageNet Accuracy':
                #ignore the ID acc term
                continue
            ood_acc += v
            num_datasets += 1
    if num_datasets != 0:
        ood_acc = ood_acc / num_datasets
    else:
        ood_acc = 0
    epoch_stats['Avg OOD Acc'] = round(ood_acc, 4)
    logger.info(f"Avg OOD Acc : {ood_acc:.4f}")
    stats.append(epoch_stats)
    stats_df = pd.DataFrame(stats)
    log_dir = "expt_logs/" + args.exp_name + "/" + "_BS" + str(
        args.batch_size) + "_WD" + str(args.wd) + "_LR" + str(args.lr) + "_run" + str(args.run)
    os.makedirs(log_dir, exist_ok=True)
    stats_df.to_csv(log_dir + '/stats.tsv', sep='\t')

    if args.save is not None:
        return model_path
