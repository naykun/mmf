# Copyright (c) Facebook, Inc. and its affiliates.
import json
from typing import List, NamedTuple

from mmf.datasets.databases.annotation_database import AnnotationDatabase


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
    timed_caption: List[TimedUtterance]
    traces: List[List[TimedPoint]]
    voice_recording: str

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
        data = []
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
        self.data = data

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
