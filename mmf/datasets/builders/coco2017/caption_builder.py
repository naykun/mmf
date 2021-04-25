# Copyright (c) Facebook, Inc. and its affiliates.
from mmf.common.registry import registry
from mmf.datasets.builders.coco2017.caption_dataset import TracedCaptionCoco2017Dataset, CVLGCoco2017Dataset
from mmf.datasets.mmf_dataset_builder import MMFDatasetBuilder


@registry.register_builder("caption_coco2017")
class CaptionCoco2017Builder(MMFDatasetBuilder):
    def __init__(
        self,
        dataset_name="caption_coco2017",
        dataset_class=TracedCaptionCoco2017Dataset,
        *args,
        **kwargs
    ):
        super().__init__(dataset_name, dataset_class, *args, **kwargs)

    @classmethod
    def config_path(cls):
        return "configs/datasets/coco2017/traced_caption.yaml"


@registry.register_builder("cvlg_coco2017")
class CaptionCoco2017Builder(MMFDatasetBuilder):
    def __init__(
        self,
        dataset_name="cvlg_coco2017",
        dataset_class=CVLGCoco2017Dataset,
        *args,
        **kwargs
    ):
        super().__init__(dataset_name, dataset_class, *args, **kwargs)

    @classmethod
    def config_path(cls):
        return "configs/datasets/coco2017/traced_caption.yaml"
