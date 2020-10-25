import torch 

from mmf.common.registry import registry
from mmf.models.base_model import BaseModel

from mmf.utils.build import (
    build_classifier_layer,
    build_image_encoder,
    build_text_encoder,
)
from transformers import BertConfig, EncoderDecoderConfig, EncoderDecoderModel

@registry.register_model("traced_encoder_decoder")
class TracedEncoderDecoder(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.build()
    
    @classmethod
    def config_path(cls):
        return "configs/models/ted/defaults.yaml"

    def build(self):

        # to be further set
        config_encoder = BertConfig()
        config_decoder = BertConfig()
        # breakpoint()
        self.image_feature_module = build_image_encoder(self.config.image_feature_processor,direct_features=True)
        self.encoderdecoder = EncoderDecoderModel.from_encoder_decoder_pretrained('bert-base-uncased', 'bert-base-uncased')

    def forward(self, sample_list, *args, **kwargs):

        decoder_input_ids = sample_list["input_ids"]
        # using default mask
        # target_mask = sample_list["input_mask"]
        segment_ids = sample_list["segment_ids"]
        token_attends = sample_list['token_attends']

        bbox_feature = sample_list['image_feature_0']

        # maybe positional encoder later
        inputs_embeds = self.image_feature_module(bbox_feature)

        decoder_output = self.encoderdecoder(
            decoder_input_ids=decoder_input_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=True,return_dict=True)

        logits = decoder_output['logits']
        decoder_attentions = decoder_output['decoder_attentions']
        
        model_output = {}
        model_output["scores"] = logits
        sample_list["targets"] = decoder_input_ids

        return model_output