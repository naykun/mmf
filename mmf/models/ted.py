import torch

from mmf.common.registry import registry
from mmf.common.sample import SampleList
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
        self.image_feature_module = build_image_encoder(
            self.config.image_feature_processor, direct_features=True)
        self.encoderdecoder = EncoderDecoderModel.from_encoder_decoder_pretrained(
            'bert-base-uncased', 'bert-base-uncased')
        self.BOS_ID = 101

    def forward(self, sample_list, *args, **kwargs):

        # breakpoint()
        decoder_input_ids = sample_list["input_ids"][:,:-1]
        # using default mask
        # target_mask = sample_list["input_mask"]
        segment_ids = sample_list["segment_ids"]
        token_attends = sample_list['token_attends']

        bbox_feature = sample_list['image_feature_0']

        # maybe positional encoder later
        inputs_embeds = self.image_feature_module(bbox_feature)
        batch_size = inputs_embeds.shape[0]

        if self.training:
            decoder_output = self.encoderdecoder(
                decoder_input_ids=decoder_input_ids,
                inputs_embeds=inputs_embeds,
                output_attentions=True, return_dict=True)

            logits = decoder_output['logits']
            decoder_attentions = decoder_output['decoder_attentions']
            cross_attentions = []
            for layer_idx in range(self.config.num_layers):
                cross_attentions.append(decoder_attentions[layer_idx][1].mean(dim=1))
            # breakpoint()

            model_output = {}
            model_output["captions"] = torch.max(logits, dim=-1)[1]
            model_output["scores"] = logits
            model_output["cross_attentions"] = cross_attentions
            sample_list["targets"] = sample_list["input_ids"][:,1:]
        else:
            generate_output = self.encoderdecoder.generate(
                input_ids=None, input_embeds=inputs_embeds, max_length=64, decoder_start_token_id=self.BOS_ID)
            model_output = {}
            model_output["captions"] = generate_output
            model_output["losses"] = {}
            loss_key = "{}/{}".format(
                sample_list.dataset_name, sample_list.dataset_type
            )
            # Add a dummy loss so that loss calculation is not required
            model_output["losses"][loss_key + "/dummy_loss"] = torch.zeros(
                batch_size, device=sample_list.image_feature_0.device
            )
            # breakpoint()

        return model_output
