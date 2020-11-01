# Copyright (c) Facebook, Inc. and its affiliates.

import json

import numpy as np

import tools.scripts.coco.coco_caption_eval as coco_caption_eval


def print_metrics(res_metrics):
    print(res_metrics)
    keys = [
        "Bleu_1",
        "Bleu_2",
        "Bleu_3",
        "Bleu_4",
        "METEOR",
        "ROUGE_L",
        "SPICE",
        "CIDEr",
    ]
    print("\n\n**********\nFinal model performance:\n**********")
    for k in keys:
        print(k, ": %.1f" % (res_metrics[k] * 100))

def pad_filter(tokenlist):
    return [token for token in tokenlist if token != "[PAD]"]

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_file", type=str, required=True)
    parser.add_argument("--annotation_file", type=str, required=True)
    parser.add_argument("--set", type=str, default="val")
    args = parser.parse_args()

    with open(args.pred_file) as f:
        preds = json.load(f)
    annotation_file = args.annotation_file
    # imdb = json.load(open(annotation_file,'r'))
    
    gts = []
    with open(annotation_file,'r') as fin:
        for line in fin:
            info = json.loads(line)
            gts.append({"image_id": info["image_id"], "caption": info["caption"]} 
            )
            # breakpoint()
    preds = [{"image_id": p["image_id"], "caption": ' '.join(pad_filter(p["caption"]))} for p in preds]
    imgids = list({g["image_id"] for g in gts})

    metrics = coco_caption_eval.calculate_metrics(
        imgids, {"annotations": gts}, {"annotations": preds}
    )

    print_metrics(metrics)
