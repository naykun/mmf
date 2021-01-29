# Copyright (c) Facebook, Inc. and its affiliates.

import random

import numpy as np
import torch
from mmf.common.registry import registry
from mmf.common.sample import Sample, SampleList
from mmf.datasets.processors.processors import BaseProcessor
from transformers.models.auto import AutoTokenizer

import nltk


nltk.download("stopwords")


@registry.register_processor("masked_token")
class MaskedTokenProcessor(BaseProcessor):
    _CLS_TOKEN = "[CLS]"
    _SEP_TOKEN = "[SEP]"

    def __init__(self, config, *args, **kwargs):
        tokenizer_config = config.tokenizer_config
        self._tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_config.type, **tokenizer_config.params
        )

        self._max_seq_length = config.max_seq_length
        self._probability = getattr(config, "mask_probability", 0.15)

    def get_vocab_size(self):
        return len(self._tokenizer)

    def tokenize(self, tokens):
        return self._tokenizer.tokenize(tokens)

    def _random_word(self, tokens, probability=0.15):
        labels = []
        for idx, token in enumerate(tokens):
            prob = random.random()

            if prob < probability:
                prob /= probability

                # 80% randomly change token to mask token
                if prob < 0.8:
                    tokens[idx] = "[MASK]"
                # 10% randomly change token to random token
                elif prob < 0.9:
                    tokens[idx] = self._tokenizer.convert_ids_to_tokens(
                        torch.randint(len(self._tokenizer), (1,), dtype=torch.long)
                    )[0]

                # rest 10% keep the original token as it is

                labels.append(self._tokenizer.convert_tokens_to_ids(token))
            else:
                labels.append(-1)

        return tokens, labels

    def _truncate_seq_pair(self, tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        if tokens_b is None:
            tokens_b = []
        else:
            # _convert_to_indices does [CLS] tokens_a [SEP] tokens_b [SEP]
            max_length -= 1
            assert max_length >= 0, (
                "Max length should be minimum 2 in case of single sentence"
                + " and 3 in case of two sentences."
            )

        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def _convert_to_indices(self, tokens_a, tokens_b=None, probability=0.15):
        tokens_a, label_a = self._random_word(tokens_a, probability=probability)
        tokens = [self._CLS_TOKEN] + tokens_a + [self._SEP_TOKEN]
        segment_ids = [0] + [0] * len(tokens_a) + [0]

        if tokens_b:
            tokens_b, label_b = self._random_word(tokens_b, probability=probability)
            lm_label_ids = [-1] + label_a + [-1] + label_b + [-1]
            assert len(tokens_b) > 0
            tokens += tokens_b + [self._SEP_TOKEN]
            segment_ids += [1] * len(tokens_b) + [1]
        else:
            lm_label_ids = [-1] + label_a + [-1]

        input_ids = self._tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < self._max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            lm_label_ids.append(-1)

        assert len(input_ids) == self._max_seq_length
        assert len(input_mask) == self._max_seq_length
        assert len(segment_ids) == self._max_seq_length
        assert len(lm_label_ids) == self._max_seq_length

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        input_mask = torch.tensor(input_mask, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)
        lm_label_ids = torch.tensor(lm_label_ids, dtype=torch.long)
        return {
            "input_ids": input_ids,
            "input_mask": input_mask,
            "segment_ids": segment_ids,
            "lm_label_ids": lm_label_ids,
            "tokens": tokens,
        }

    def __call__(self, item):
        text_a = item["text_a"]
        text_b = item.get("text_b", None)

        tokens_a = self.tokenize(text_a)
        tokens_b = None

        if text_b:
            tokens_b = self.tokenize(text_b)

        self._truncate_seq_pair(tokens_a, tokens_b, self._max_seq_length - 2)
        output = self._convert_to_indices(
            tokens_a, tokens_b, probability=self._probability
        )
        output["is_correct"] = torch.tensor(item["is_correct"], dtype=torch.long)

        return output


@registry.register_processor("bert_tokenizer")
class BertTokenizer(MaskedTokenProcessor):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self._probability = 0

    def __call__(self, item):

        if "text" in item:
            text_a = item["text"]
        else:
            text_a = " ".join(item["tokens"])

        if isinstance(text_a, list):
            text_a = " ".join(text_a)

        tokens_a = self.tokenize(text_a)

        # 'text_b' can be defined in the dataset preparation
        tokens_b = None
        if "text_b" in item:
            text_b = item["text_b"]
            if text_b:
                tokens_b = self.tokenize(text_b)

        self._truncate_seq_pair(tokens_a, tokens_b, self._max_seq_length - 2)
        output = self._convert_to_indices(
            tokens_a, tokens_b, probability=self._probability
        )
        output["text"] = output["tokens"]
        return output


@registry.register_processor("multi_sentence_bert_tokenizer")
class MultiSentenceBertTokenizer(BertTokenizer):
    """Extension of BertTokenizer which supports multiple sentences.
    Separate from normal usecase, each sentence will be passed through
    bert tokenizer separately and indices will be reshaped as single
    tensor. Segment ids will also be increasing in number.
    """

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.fusion_strategy = config.get("fusion", "concat")
        self.tokenizer = super().__call__

    def __call__(self, item):
        texts = item["text"]
        if not isinstance(texts, list):
            texts = [texts]

        processed = []
        for idx, text in enumerate(texts):
            sample = Sample()
            processed_text = self.tokenizer({"text": text})
            sample.update(processed_text)
            sample.segment_ids.fill_(idx)
            processed.append(sample)
        # Use SampleList to convert list of tensors to stacked tensors
        processed = SampleList(processed)
        if self.fusion_strategy == "concat":
            processed.input_ids = processed.input_ids.view(-1)
            processed.input_mask = processed.input_mask.view(-1)
            processed.segment_ids = processed.segment_ids.view(-1)
            processed.lm_label_ids = processed.lm_label_ids.view(-1)
        return processed.to_dict()


@registry.register_processor("traced_bert_tokenizer")
class TracedBertTokenizer(MaskedTokenProcessor):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self._probability = 0
        self.segment_reverse = (
            config.segment_reverse if hasattr(config, "segment_reverse") else False
        )
        self.sync_seg_reverse = (
            config.sync_seg_reverse if hasattr(config, "sync_seg_reverse") else False
        )
        self.sync_seg_shuffle = (
            config.sync_seg_shuffle if hasattr(config, "sync_seg_shuffle") else False
        )
        self.filter_vocab = (
            config.filter_vocab if hasattr(config, "filter_vocab") else "none"
        )
        registry.register("ln_caption_processor", self)

        # attention guidance stop word / filter list
        from nltk.corpus import stopwords

        # import nltk

        # nltk.download("stopwords")
        self.stop_words = set(stopwords.words("english"))
        for w in ["!", ",", ".", "?", "-s", "-ly", "</s>", "s"]:
            self.stop_words.add(w)
        COCO_CATE = [
            "person",
            "bicycle",
            "car",
            "motorcycle",
            "airplane",
            "bus",
            "train",
            "truck",
            "boat",
            "traffic light",
            "fire hydrant",
            "stop sign",
            "parking meter",
            "bench",
            "bird",
            "cat",
            "dog",
            "horse",
            "sheep",
            "cow",
            "elephant",
            "bear",
            "zebra",
            "giraffe",
            "backpack",
            "umbrella",
            "handbag",
            "tie",
            "suitcase",
            "frisbee",
            "skis",
            "snowboard",
            "sports ball",
            "kite",
            "baseball bat",
            "baseball glove",
            "skateboard",
            "surfboard",
            "tennis racket",
            "bottle",
            "wine glass",
            "cup",
            "fork",
            "knife",
            "spoon",
            "bowl",
            "banana",
            "apple",
            "sandwich",
            "orange",
            "broccoli",
            "carrot",
            "hot dog",
            "pizza",
            "donut",
            "cake",
            "chair",
            "couch",
            "potted plant",
            "bed",
            "dining table",
            "toilet",
            "tv",
            "laptop",
            "mouse",
            "remote",
            "keyboard",
            "cell phone",
            "microwave",
            "oven",
            "toaster",
            "sink",
            "refrigerator",
            "book",
            "clock",
            "vase",
            "scissors",
            "teddy bear",
            "hair drier",
            "toothbrush",
        ]
        self.coco_vocab = set()
        for w in COCO_CATE:
            piece = self.tokenize(w)
            for p in piece:
                self.coco_vocab.add(p)

        from mmf.utils.download import download
        import os
        import random

        path = "/tmp/"
        filename = "VG_categoty{}.txt".format(random.randint(0, 10000))
        filepath = os.path.join(path, filename)
        url = "http://visualgenome.org/static/data/dataset/object_alias.txt"
        if download(url, path, filename, redownload=False):
            cate = []
            with open(filepath) as fin:
                for line in fin:
                    cate += line.strip().split(",")
            self.vg_vocab = set()
            for w in cate:
                piece = self.tokenize(w)
                for p in piece:
                    self.vg_vocab.add(p)

    def __call__(self, item):

        timed_caption = item["timed_caption"]
        bbox_attend_scores = item["bbox_attend_scores"]
        # breakpoint()
        attend_len = bbox_attend_scores.shape[0]
        tokens = []
        token_attends = []
        seg_ids = None
        # print(bbox_attend_scores.shape)
        # print(len(timed_caption))

        seg_indices = []
        for i, word in enumerate(timed_caption):
            text = word["utterance"]
            # wired length mis-matching
            attend = bbox_attend_scores[min(i, attend_len - 1)]
            token = self.tokenize(text)
            tokens += token
            token_attends += [attend] * len(token)
            if "." in text:
                seg_indices.append(len(tokens))
        # guard for sentence without "."
        if len(seg_indices) == 0 or seg_indices[-1] != len(tokens):
            seg_indices.append(len(tokens))

        if self.sync_seg_reverse:
            import random

            sync_reverse = random.random() > 0.5

        if self.segment_reverse or (self.sync_seg_reverse and sync_reverse):
            seg_start = [0] + seg_indices[:-1]
            seg_end = seg_indices
            seg_s_e = list(zip(seg_start, seg_end))
            # print(seg_s_e)
            tokens_segs = [tokens[s:e] for s, e in seg_s_e]
            token_attends_segs = [token_attends[s:e] for s, e in seg_s_e]

            if self.sync_seg_shuffle:
                shuffle_order = list(range(len(tokens_segs)))
                random.shuffle(shuffle_order)
                tokens_segs = [tokens_segs[i] for i in shuffle_order]
                token_attends_segs = [token_attends_segs[i] for i in shuffle_order]
            else:
                tokens_segs.reverse()
                token_attends_segs.reverse()

            tokens = [token for seg in tokens_segs for token in seg]
            token_attends = [attend for seg in token_attends_segs for attend in seg]
            # start from 1
            seg_ids = [i + 1 for i, seg in enumerate(tokens_segs) for token in seg]
            # import ipdb; ipdb.set_trace()

        tokens = tokens[: self._max_seq_length - 1]
        token_attends = token_attends[: self._max_seq_length - 1]
        if seg_ids is None:
            seg_ids = [1] * len(tokens)
        seg_ids = seg_ids[: self._max_seq_length - 1]

        # generate attention supervision mask
        # import ipdb; ipdb.set_trace()
        if self.filter_vocab == "stopword":
            attend_masklist = [0 if tt in self.stop_words else 1 for tt in tokens]
        elif self.filter_vocab == "coco":
            attend_masklist = [
                1 if (tt not in self.stop_words and tt in self.coco_vocab) else 0
                for tt in tokens
            ]
        elif self.filter_vocab == "vg":
            attend_masklist = [
                1 if (tt not in self.stop_words and tt in self.vg_vocab) else 0
                for tt in tokens
            ]
        elif self.filter_vocab == "none":
            attend_masklist = [1] * len(tokens)

        output = self._convert_to_indices(
            tokens, token_attends, seg_ids, attend_masklist
        )
        if self.sync_seg_reverse:
            output["sync_reverse"] = sync_reverse
            if self.sync_seg_shuffle and sync_reverse:
                output["sync_shuffle_order"] = shuffle_order
            else:
                output["sync_shuffle_order"] = None
        return output

    def _convert_to_indices(self, tokens, token_attends, seg_ids, attend_mask):
        tokens = [self._CLS_TOKEN] + tokens
        token_attends = [np.zeros_like(token_attends[0])] + token_attends
        # attend_length = len(token_attends[0])
        segment_ids = [0] + seg_ids
        attend_mask = [0] + attend_mask

        input_ids = self._tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < self._max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            token_attends.append(np.zeros_like(token_attends[0]))
            attend_mask.append(0)

        assert len(input_ids) == self._max_seq_length
        assert len(input_mask) == self._max_seq_length
        assert len(segment_ids) == self._max_seq_length
        assert len(token_attends) == self._max_seq_length
        assert len(attend_mask) == self._max_seq_length

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        input_mask = torch.tensor(input_mask, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)
        token_attends = torch.tensor(token_attends, dtype=torch.float)
        attend_mask = torch.tensor(attend_mask, dtype=torch.long)
        return {
            "input_ids": input_ids,
            "input_mask": input_mask,
            "segment_ids": segment_ids,
            "token_attends": token_attends,
            "text": tokens,
            "attend_supervision_mask": attend_mask,
        }

    def id2tokens(self, ids):
        return self._tokenizer.decode(ids, skip_special_tokens=True)

    def id2rawtoken(self, ids):
        return self._tokenizer.convert_ids_to_tokens(ids)


@registry.register_processor("spatial_trace_tokenizer")
class SpatialTraceTokenizer(BaseProcessor):
    def __init__(self, config, *args, **kwargs):
        self._max_seq_length = config.max_seq_length
        self.delta = config.delta
        self.reverse = config.reverse
        self.segment_reverse = (
            config.segment_reverse if hasattr(config, "segment_reverse") else False
        )
        self.sync_seg_reverse = (
            config.sync_seg_reverse if hasattr(config, "sync_seg_reverse") else False
        )
        self.time_window = config.time_window if hasattr(config, "time_window") else 0.4

    def __call__(
        self, image_info_0, sample_info, sync_reverse=False, sync_shuffle_order=None
    ):
        # h, w = (image_info_0["image_height"], image_info_0["image_width"])
        traces = [x for tr in sample_info["traces"] for x in tr]

        current_t = 0
        current_trace_window = []
        trace_boxes = []
        for t in traces:
            if t["t"] > current_t:
                current_t += self.time_window
                if len(current_trace_window) > 0:
                    points = np.array(current_trace_window)
                    x1, y1 = points.min(axis=0) * (1 - self.delta)
                    x2, y2 = points.max(axis=0) * (1 + self.delta)
                    area = (x2 - x1) * (y2 - y1)
                    trace_boxes.append([x1, y1, x2, y2, area, t["t"]])
                    current_trace_window = []
            else:
                current_trace_window.append([t["x"], t["y"]])
        if self.segment_reverse or (self.sync_seg_reverse and sync_reverse):
            timed_caption = sample_info["timed_caption"]
            time_slot = []
            for utter in timed_caption:
                if "." in utter["utterance"]:
                    time_slot.append(utter["end_time"])
            segments = []
            segment = []
            seg_id = 0
            for box in trace_boxes:
                if seg_id < len(time_slot) and box[-1] > time_slot[seg_id]:
                    seg_id += 1
                    segments.append(segment)
                    segment = []
                else:
                    segment.append(box[:-1])
            if len(segment) > 0:
                segments.append(segment)
                segment = []
            if sync_shuffle_order is not None:
                max_segments_id = len(segments) - 1
                # print(len_segments)
                # print(sync_shuffle_order)
                if max_segments_id >= 0:
                    segments = [
                        segments[min(i, max_segments_id)] for i in sync_shuffle_order
                    ]
            else:
                segments.reverse()
            trace_boxes = [box for seg in segments for box in seg]
            seg_id = [i + 1 for i, seg in enumerate(segments) for box in seg]
        else:
            trace_boxes = [box[:-1] for box in trace_boxes]
            seg_id = [1] * len(trace_boxes)
        trace_boxes, trace_boxes_mask, boxes_seg_id, contr_seg_id = self._trancate(
            trace_boxes, seg_id
        )
        trace_boxes = torch.tensor(trace_boxes, dtype=torch.float)
        trace_boxes_mask = torch.tensor(trace_boxes_mask, dtype=torch.long)
        boxes_seg_id = torch.tensor(boxes_seg_id, dtype=torch.long)
        contr_seg_id = torch.tensor(contr_seg_id, dtype=torch.long)
        return {
            "trace_boxes": trace_boxes,
            "trace_boxes_mask": trace_boxes_mask,
            "trace_boxes_seg_id": boxes_seg_id,
            "trace_boxes_loop_contrastive_seg_id": contr_seg_id,
        }

    def _trancate(self, boxes, seg_id):
        boxes = boxes[: self._max_seq_length]
        seg_id = seg_id[: self._max_seq_length]
        if self.reverse and not self.segment_reverse:
            boxes.reverse()
        num_boxes = len(boxes)
        appendix = [[0.0] * 5] * (self._max_seq_length - num_boxes)
        boxes += appendix
        box_mask = [1] * num_boxes + [0] * (self._max_seq_length - num_boxes)
        loop_contrastive_seg_id = seg_id + [0] * (self._max_seq_length - num_boxes)
        box_seg_id = [1] * self._max_seq_length
        return boxes, box_mask, box_seg_id, loop_contrastive_seg_id
