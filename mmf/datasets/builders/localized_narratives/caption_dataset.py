from abc import ABC


from mmf.common.sample import Sample
from mmf.common.typings import MMFDatasetConfigType
from mmf.datasets.builders.localized_narratives.database import (
    LocalizedNarrativesAnnotationDatabase, BboxAlignedLocalizedNarrativesAnnotationDatabase
)
from mmf.datasets.mmf_dataset import MMFDataset


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
            current_sample.update(features)

        # breakpoint()
        processed_caption = self.caption_processor(
            {"timed_caption": sample_info["timed_caption"], "bbox_attend_scores": image_info_0['bbox_attend_scores']})
        # should be a trace enhanced processor
        current_sample.update(processed_caption)
        current_sample.image_id = sample_info["image_id"]
        current_sample.feature_path = sample_info["feature_path"]

        return current_sample


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
