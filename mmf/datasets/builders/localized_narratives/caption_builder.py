# Copyright (c) Facebook, Inc. and its affiliates.
from mmf.common.registry import registry
from mmf.datasets.builders.localized_narratives.caption_dataset import (
    TracedCaptionLocalizedNarrativesDataset)
from mmf.datasets.mmf_dataset_builder import MMFDatasetBuilder


@registry.register_builder("caption_localized_narratives")
class CaptionLocalizedNarrativesBuilder(MMFDatasetBuilder):
    def __init__(
        self,
        dataset_name="caption_localized_narratives",
        dataset_class=TracedCaptionLocalizedNarrativesDataset,
        *args,
        **kwargs
    ):
        super().__init__(dataset_name, dataset_class, *args, **kwargs)

    @classmethod
    def config_path(cls):
        return "configs/datasets/localized_narratives/traced_caption.yaml"
