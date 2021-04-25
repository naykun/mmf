# Copyright (c) Facebook, Inc. and its affiliates.

import json
import os

import coco_caption_eval


# import numpy as np


def print_metrics(res_metrics):
    print(res_metrics)
    keys = [
        "Bleu_1",
        "Bleu_2",
        "Bleu_3",
        "Bleu_4",
        "METEOR",
        "ROUGE_L",
        "ROUGE_F1",
        "SPICE",
        "CIDEr",
    ]
    print("\n\n**********\nFinal model performance:\n**********")
    for k in keys:
        print(k, ": %.1f" % (res_metrics[k] * 100))


def dump_to_json(output_path, res_metrics):
    filepath = os.path.join(output_path, "metric.json")
    json.dump(res_metrics, open(filepath, "w"))


def pad_filter(tokenlist):
    return [token for token in tokenlist if token != "[PAD]"]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", type=str, required=True)
    parser.add_argument("--annotation_file", type=str, required=True)
    parser.add_argument("--set", type=str, default="val")
    parser.add_argument("--json-dir", default=os.environ.get("PT_OUTPUT_DIR", "save"))
    args = parser.parse_args()

    preds = None
    for p, _, fn in os.walk(args.pred_dir):
        print(fn)
        if (
            len(fn) > 0
            and "caption_coco2017_run_test" in fn[0]
            and fn[0].endswith(".json")
        ):
            with open(os.path.join(p, fn[0])) as f:
                # print(fn)
                preds = json.load(f)
            # break
    annotation_file = args.annotation_file
    # imdb = json.load(open(annotation_file,'r'))

    gts = []
    with open(annotation_file) as fin:
        for i, line in enumerate(fin):
            info = json.loads(line)
            caps = info["caption"].split(".")
            for j, cap in enumerate(caps):
                gts.append(
                    {"image_id": info["image_id"] + "-" + str(i), "caption": cap}
                )
    # breakpoint()
    preds = [
        {"image_id": p["image_id"] + "-" + str(i), "caption": cap}
        for i, p in enumerate(preds)
        for j, cap in enumerate(" ".join(pad_filter(p["caption"])).split("."))
    ]
    imgids = list({g["image_id"] for g in gts})

    # import ipdb; ipdb.set_trace()
    metrics = coco_caption_eval.calculate_metrics(
        imgids, {"annotations": gts}, {"annotations": preds}
    )

    print_metrics(metrics)
    dump_to_json(args.json_dir, metrics)
