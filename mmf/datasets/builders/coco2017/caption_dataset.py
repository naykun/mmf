# Copyright (c) Facebook, Inc. and its affiliates.

from mmf.common.typings import MMFDatasetConfigType
from mmf.datasets.builders.localized_narratives.caption_dataset import (
    TracedCaptionLocalizedNarrativesDatasetMixin,
    CVLGLocalizedNarrativesDatasetMixin
)
from mmf.datasets.mmf_dataset import MMFDataset


class TracedCaptionCoco2017Dataset(
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
            "caption_coco2017", config, dataset_type, index, *args, **kwargs
        )

class CVLGCoco2017Dataset(
    CVLGLocalizedNarrativesDatasetMixin, MMFDataset
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
            "cvlg_coco2017", config, dataset_type, index, *args, **kwargs
        )
