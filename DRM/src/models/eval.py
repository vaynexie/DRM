import os
import json

import torch
import numpy as np
from src.models import utils
from src.datasets.common import get_dataloader, maybe_dictionarize
import src.datasets as datasets
import torch.nn.functional as F


def eval_single_dataset(image_classifier, dataset, args, classification_head_llm,classification_head_basic,beta=0.5):

    model = image_classifier
    input_key = 'images'
    image_enc = None

    model.eval()
    classification_head_llm.eval()
    classification_head_basic.eval()

    dataloader = get_dataloader(dataset,
                                is_train=False,
                                args=args,
                                image_encoder=image_enc)

    batched_data = enumerate(dataloader)
    device = args.device

    if hasattr(dataset, 'post_loop_metrics'):
        # keep track of labels, predictions and metadata
        all_labels, all_preds, all_metadata = [], [], []

    with torch.no_grad():
        top1, correct, n = 0., 0., 0.
        for i, data in batched_data:

            data = maybe_dictionarize(data)
            x = data[input_key].to(device)
            y = data['labels'].to(device)
            #print(x.shape)

            if 'image_paths' in data:
                image_paths = data['image_paths']

            logits_llm = utils.get_logits(x, model, classification_head_llm)
            logits_basic = utils.get_logits(x, model, classification_head_basic)
            projection_fn = getattr(dataset, 'project_logits', None)
            if projection_fn is not None:
                logits_llm = projection_fn(logits_llm, device)
                logits_basic = projection_fn(logits_basic, device)

            probabilities_llm = F.softmax(logits_llm, dim=1)
            probabilities_basic = F.softmax(logits_basic, dim=1)
            beta=float(beta)
            probabilities = beta*probabilities_basic+(1-beta)*probabilities_llm

            if hasattr(dataset, 'project_labels'):
                y = dataset.project_labels(y, device)

            #pred = logits.argmax(dim=1, keepdim=True).to(device)
            pred = probabilities.argmax(dim=1, keepdim=True).to(device)

            if hasattr(dataset, 'accuracy'):
                # acc1, num_total = dataset.accuracy(logits, y, image_paths,
                #                                    args)
                acc1, num_total = dataset.accuracy(probabilities, y, image_paths,
                                                   args)
                correct += acc1
                n += num_total
            else:
                correct += pred.eq(y.view_as(pred)).sum().item()
                n += y.size(0)

            if hasattr(dataset, 'post_loop_metrics'):
                all_labels.append(y.cpu().clone().detach())
                # all_preds.append(logits.cpu().clone().detach())
                all_preds.append(probabilities.cpu().clone().detach())
                metadata = data[
                    'metadata'] if 'metadata' in data else image_paths
                all_metadata.extend(metadata)

        top1 = correct / n

        if hasattr(dataset, 'post_loop_metrics'):
            all_labels = torch.cat(all_labels)
            all_preds = torch.cat(all_preds)
            metrics = dataset.post_loop_metrics(all_labels, all_preds,
                                                all_metadata, args)
            # metrics = dataset.post_loop_metrics(all_labels.cpu().clone().detach().numpy(), all_preds.cpu().clone().detach().numpy(),
            #                                     all_metadata, args)
            if 'acc' in metrics:
                metrics['top1'] = metrics['acc']
        else:
            metrics = {}
    if 'top1' not in metrics:
        metrics['top1'] = top1

    del logits_llm
    del logits_basic
    del probabilities_llm
    del probabilities_basic
    del probabilities
    del all_preds
    del pred

    return metrics


def eval_single_batch_dataset(image_classifier, dataset, args,
                              classification_head, data):

    model = image_classifier
    input_key = 'images'

    model.eval()
    classification_head.eval()

    device = args.device

    if hasattr(dataset, 'post_loop_metrics'):
        # keep track of labels, predictions and metadata
        all_labels, all_preds, all_metadata = [], [], []

    with torch.no_grad():
        top1, correct, n, cnt_loss = 0., 0., 0., 0.

        data = maybe_dictionarize(data)
        x = data[input_key].to(device)
        y = data['labels'].to(device)

        assert x.shape[0] == 2 * args.k, 'val mismatch size'

        if 'image_paths' in data:
            image_paths = data['image_paths']

        logits = utils.get_logits(x, model, classification_head)

        projection_fn = getattr(dataset, 'project_logits', None)
        if projection_fn is not None:
            logits = projection_fn(logits, device)

        if hasattr(dataset, 'project_labels'):
            y = dataset.project_labels(y, device)

        cnt_loss = F.cross_entropy(logits, y)
        pred = logits.argmax(dim=1, keepdim=True).to(device)
        if hasattr(dataset, 'accuracy'):
            acc1, num_total = dataset.accuracy(logits, y, image_paths, args)
            correct += acc1
            n += num_total
        else:
            correct += pred.eq(y.view_as(pred)).sum().item()
            n += y.size(0)

        if hasattr(dataset, 'post_loop_metrics'):
            all_labels.append(y.cpu().clone().detach())
            all_preds.append(logits.cpu().clone().detach())
            metadata = data['metadata'] if 'metadata' in data else image_paths
            all_metadata.extend(metadata)

        top1 = correct / n

        if hasattr(dataset, 'post_loop_metrics'):
            all_labels = torch.cat(all_labels)
            all_preds = torch.cat(all_preds)
            metrics = dataset.post_loop_metrics(all_labels, all_preds,
                                                all_metadata, args)
            if 'acc' in metrics:
                metrics['top1'] = metrics['acc']
        else:
            metrics = {}
    if 'top1' not in metrics:
        metrics['top1'] = top1

    return metrics['top1'], cnt_loss.item()


def evaluate(image_classifier,
             args,
             classification_head_llm,
             classification_head_basic,
             beta=0.5,
             train_stats={},
             logger=None):
    if args.eval_datasets is None:
        return
    info = vars(args)
    for i, dataset_name in enumerate(args.eval_datasets):
        print('Evaluating on', dataset_name)
        dataset_class = getattr(datasets, dataset_name)
        dataset = dataset_class(image_classifier.module.val_preprocess,
                                location=args.data_location,
                                batch_size=args.batch_size)

        results = eval_single_dataset(image_classifier, dataset, args,
                                      classification_head_llm,classification_head_basic,beta=beta)

        if 'top1' in results:
            print(f"{dataset_name} Top-1 accuracy: {results['top1']:.4f}")
            if logger != None:
                logger.info(
                    f"{dataset_name} Top-1 accuracy: {results['top1']:.4f}")
            train_stats[dataset_name + " Accuracy"] = round(results['top1'], 4)

        for key, val in results.items():
            if 'worst' in key or 'f1' in key.lower() or 'pm0' in key:
                print(f"{dataset_name} {key}: {val:.4f}")
                if logger != None:
                    logger.info(f"{dataset_name} {key}: {val:.4f}")
                train_stats[dataset_name + key] = round(val, 4)

    return info
