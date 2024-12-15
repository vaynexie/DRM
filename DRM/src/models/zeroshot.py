import os
import torch
from tqdm import tqdm
import numpy as np
import clip_m.clip as clip
import src.templates as templates
import src.datasets as datasets
from src.args import parse_arguments
from src.models.modeling import ClassificationHead, CLIPEncoder, ImageClassifier



def get_zeroshot_classifier(args, clip_model):
    import json
    with open(args.cd_path) as json_file:
        cd_data = json.load(json_file)
    cd_data_key=list(cd_data.keys())
    assert args.template is not None
    assert args.train_dataset is not None
    template = getattr(templates, args.template)
    logit_scale = clip_model.logit_scale

    few_shot_data_list = ["ImageNetKShot", "PatchCamelyonVal"]
    dataset_class = getattr(datasets, args.train_dataset)
    if args.train_dataset in few_shot_data_list:
        print(f"Doing {args.k} shot classification")
        dataset = dataset_class(None,
                                location=args.data_location,
                                batch_size=args.batch_size,
                                k=args.k)
    else:
        dataset = dataset_class(None,
                                location=args.data_location,
                                batch_size=args.batch_size)
    device = args.device
    clip_model.eval()
    clip_model.to(device)

    kkk=0
    with torch.no_grad():
        zeroshot_weights_llm = []
        ## get concept description prompts
        for classname in tqdm(dataset.classnames):
            texts_llm = []
            texts_llm.append(cd_data[cd_data_key[kkk]])
            texts_llm = clip.tokenize(texts_llm).to(device)  # tokenize
            embeddings_llm = clip_model.encode_text(
                texts_llm)  # embed with text encoder
            embeddings_llm /= embeddings_llm.norm(dim=-1, keepdim=True)
            embeddings_llm = embeddings_llm.mean(dim=0, keepdim=True)
            embeddings_llm /= embeddings_llm.norm()

            zeroshot_weights_llm.append(embeddings_llm)
            kkk+=1

        zeroshot_weights_llm = torch.stack(zeroshot_weights_llm, dim=0).to(device)
        zeroshot_weights_llm = torch.transpose(zeroshot_weights_llm, 0, 2)
        zeroshot_weights_llm *= logit_scale.exp()
        zeroshot_weights_llm = zeroshot_weights_llm.squeeze().float()
        zeroshot_weights_llm = torch.transpose(zeroshot_weights_llm, 0, 1)


    classification_head_llm = ClassificationHead(normalize=True,
                                             weights=zeroshot_weights_llm)

    del texts_llm,embeddings_llm,zeroshot_weights_llm

    with torch.no_grad():
        zeroshot_weights_basic = []
        ## get default prompts
        for classname in tqdm(dataset.classnames):
            texts_basic = []
            #*********************************
            for t in template:
                texts_basic.append(t(classname))
            #*********************************
            texts_basic = clip.tokenize(texts_basic).to(device)  # tokenize
            embeddings_basic = clip_model.encode_text(
                texts_basic)  # embed with text encoder
            embeddings_basic /= embeddings_basic.norm(dim=-1, keepdim=True)
            embeddings_basic = embeddings_basic.mean(dim=0, keepdim=True)
            embeddings_basic /= embeddings_basic.norm()
            zeroshot_weights_basic.append(embeddings_basic)
            kkk+=1

        zeroshot_weights_basic = torch.stack(zeroshot_weights_basic, dim=0).to(device)
        zeroshot_weights_basic = torch.transpose(zeroshot_weights_basic, 0, 2)
        zeroshot_weights_basic *= logit_scale.exp()
        zeroshot_weights_basic = zeroshot_weights_basic.squeeze().float()
        zeroshot_weights_basic = torch.transpose(zeroshot_weights_basic, 0, 1)

    classification_head_basic = ClassificationHead(normalize=True,
                                             weights=zeroshot_weights_basic)

    del texts_basic,embeddings_basic,zeroshot_weights_basic

    return classification_head_llm,classification_head_basic
