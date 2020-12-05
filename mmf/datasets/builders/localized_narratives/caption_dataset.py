from abc import ABC
import torch
import numpy as np

from mmf.common.sample import Sample
from mmf.common.typings import MMFDatasetConfigType
from mmf.datasets.builders.localized_narratives.database import (
    LocalizedNarrativesAnnotationDatabase, BboxAlignedLocalizedNarrativesAnnotationDatabase
)
from mmf.datasets.mmf_dataset import MMFDataset
from mmf.utils.distributed import byte_tensor_to_object, object_to_byte_tensor


class TracedCaptionLocalizedNarrativesDatasetMixin(ABC):
    def build_annotation_db(self) -> LocalizedNarrativesAnnotationDatabase:
        annotation_path = self._get_path_based_on_index(
            self.config, "annotations", self._index
        )
        return LocalizedNarrativesAnnotationDatabase(self.config, annotation_path)

    def __getitem__(self, idx: int) -> Sample:
        sample_info = self.annotation_db[idx]
        current_sample = Sample()

        # Get the image features
        if self._use_features:
            features = self.features_db[idx]
            image_info_0 = features["image_info_0"]
            if image_info_0 and "image_id" in image_info_0.keys():
                image_info_0["feature_path"] = image_info_0["image_id"]
                image_info_0.pop("image_id")
            self.transformer_bbox_processor(features['image_info_0'])
            current_sample.update(features)

        # breakpoint()
        processed_traces = self.trace_bbox_processor(image_info_0, sample_info)
        current_sample.update(processed_traces)

        processed_caption = self.caption_processor(
            {"timed_caption": sample_info["timed_caption"], "bbox_attend_scores": image_info_0['bbox_attend_scores']})
        # should be a trace enhanced processor
        current_sample.update(processed_caption)
        current_sample.image_id = object_to_byte_tensor(sample_info["image_id"])
        current_sample.feature_path = sample_info["feature_path"]

        return current_sample

    def format_for_prediction(self, report):
        captions = report.captions.tolist()
        cross_attentions = report.cross_attention.tolist()
        predictions = []

        for idx, image_id in enumerate(report.image_id):
            image_id = byte_tensor_to_object(image_id)
            cross_attention = cross_attentions[idx]
            caption = self.caption_processor.id2tokens(captions[idx]).split()
            raw_caption = self.caption_processor.id2rawtoken(captions[idx])
            if isinstance(image_id, torch.Tensor):
                image_id = image_id.item()
            predictions.append({"image_id": image_id, "caption": caption, "cross_attention": cross_attention, "raw_caption" : raw_caption})

        return predictions


class TracedCaptionLocalizedNarrativesDataset(
    TracedCaptionLocalizedNarrativesDatasetMixin, MMFDataset
):
    def __init__(
        self,
        config: MMFDatasetConfigType,
        dataset_type: str,
        index: int,
        *args,
        **kwargs,
    ):
        super().__init__(
            "caption_localized_narratives", config, dataset_type, index, *args, **kwargs
        )
