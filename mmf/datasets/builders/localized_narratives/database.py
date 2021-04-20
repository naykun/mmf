# Copyright (c) Facebook, Inc. and its affiliates.
import json
from typing import List, NamedTuple, Optional

from mmf.datasets.databases.annotation_database import AnnotationDatabase
import lmdb
import pickle
import os

class TimedPoint(NamedTuple):
    x: float
    y: float
    t: float


class TimedUtterance(NamedTuple):
    utterance: str
    start_time: float
    end_time: float


class LocalizedNarrative(NamedTuple):
    dataset_id: str
    image_id: str
    annotator_id: int
    caption: str
    timed_caption: Optional[List[TimedUtterance]] = None
    traces: Optional[List[List[TimedPoint]]] = None
    voice_recording: Optional[str] = None

    def __repr__(self):
        truncated_caption = (
            self.caption[:60] + "..." if len(self.caption) > 63 else self.caption
        )
        truncated_timed_caption = self.timed_caption[0].__str__()
        truncated_traces = self.traces[0][0].__str__()
        return (
            f"{{\n"
            f" dataset_id: {self.dataset_id},\n"
            f" image_id: {self.image_id},\n"
            f" annotator_id: {self.annotator_id},\n"
            f" caption: {truncated_caption},\n"
            f" timed_caption: [{truncated_timed_caption}, ...],\n"
            f" traces: [[{truncated_traces}, ...], ...],\n"
            f" voice_recording: {self.voice_recording}\n"
            f"}}"
        )


class LocalizedNarrativesAnnotationDatabase(AnnotationDatabase):
    def __init__(self, config, path, *args, **kwargs):
        super().__init__(config, path, *args, **kwargs)


    def load_annotation_db(self, path):
        # import ipdb; ipdb.set_trace()
        data = []
        if path.endswith(".jsonl"):
            self.store_type = "jsonl"
            with open(path) as f:
                for line in f:
                    annotation = json.loads(line)
                    loc_narr = LocalizedNarrative(**annotation)
                    data.append(
                        {
                            "dataset_id": loc_narr.dataset_id,
                            "image_id": loc_narr.image_id,
                            "caption": loc_narr.caption,
                            "feature_path": self._feature_path(
                                loc_narr.dataset_id, loc_narr.image_id
                            ),
                            "timed_caption": loc_narr.timed_caption,
                            "traces": loc_narr.traces,
                        }
                    )
        elif path.endswith(".lmdb"):
            self.store_type = "lmdb"
            self.lmdb_path = path
            self.lmdb_env = None
            env = lmdb.open(
                path,
                subdir=os.path.isdir(path),
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
            )
            with env.begin(write=False, buffers=True) as txn:
                data = list(pickle.loads(txn.get(b"keys")))
        self.data = data
    def init_env(self):
        self.lmdb_env = lmdb.open(
            self.lmdb_path,
            subdir=os.path.isdir(self.lmdb_path),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
    def __getitem__(self, idx):
        data = self.data[idx + self.start_idx]
        if self.store_type == "lmdb":
            if self.lmdb_env is None:
                self.init_env()
            with self.lmdb_env.begin(write=False, buffers=True) as txn:
                data = pickle.loads(txn.get(data))
                loc_narr = LocalizedNarrative(**data)
                data={
                        "dataset_id": loc_narr.dataset_id,
                        "image_id": loc_narr.image_id,
                        "caption": loc_narr.caption,
                        "feature_path": self._feature_path(
                            loc_narr.dataset_id, loc_narr.image_id
                        ),
                        "timed_caption": loc_narr.timed_caption,
                        "traces": loc_narr.traces,
                    }
                # import ipdb; ipdb.set_trace()
        return data

    def _feature_path(self, dataset_id, image_id):
        if "mscoco" in dataset_id.lower():
            return image_id.rjust(12, "0") + ".npy"

        return image_id + ".npy"


# deprecated
class BboxAlignedLocalizedNarrativesAnnotationDatabase(AnnotationDatabase):
    def __init__(self, config, path, *args, **kwargs):
        super().__init__(config, path, *args, **kwargs)

    def load_annotation_db(self, path):
        data = []
        # with open(path) as f:
        #     for line_num, line in enumerate(f):
        #         # not work
        #         need_keys = ["dataset_id", "image_id", "caption"]
        #         loc_narr = {}
        #         print(line_num,end='\r')
        #         for k,v in ijson.kvitems(line,""):
        #             if k in need_keys:
        #                 loc_narr[k] = v
        #             elif k=="timed_caption":
        #                 attend_score = []
        #                 for word in v:
        #                     attend_score.append(word["bbox_attend_scores"])
        #                     # del word["bbox_attend_scores"]
        #                 loc_narr["attend_scores"] = numpy.array(attend_score,dtype=float)
        #                 loc_narr["timed_caption"] = v

        #         loc_narr["feature_path"] = self._feature_path(loc_narr["dataset_id"],loc_narr["image_id"])
        #         data.append(loc_narr)
        self.data = data

    def _feature_path(self, dataset_id, image_id):
        if "mscoco" in dataset_id.lower():
            return image_id.rjust(12, "0") + ".npy"

        return image_id + ".npy"
