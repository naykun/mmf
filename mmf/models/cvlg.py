# Copyright (c) Facebook, Inc. and its affiliates.

import torch
from mmf.common.registry import registry
from mmf.models.base_model import BaseModel
from mmf.utils.build import build_encoder, build_image_encoder
from transformers import BertConfig, EncoderDecoderConfig, EncoderDecoderModel

from axial_positional_embedding import AxialPositionalEmbedding
from dalle_pytorch import OpenAIDiscreteVAE


@registry.register_model("cvlg")
class CrossVLGenerator(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.build()

    @classmethod
    def config_path(cls):
        return "configs/models/cvlg/defaults.yaml"

    def build(self):

        # to be further set
        # breakpoint()
        self.image_feature_module = build_image_encoder(
            self.config.image_feature_processor, direct_features=True
        )
        if self.config.concate_trace:
            self.trace_feature_module = build_encoder(self.config.trace_feature_encoder)

        if self.config.base_model_name == "bert-base-uncased":
            self.encoderdecoder = EncoderDecoderModel.from_encoder_decoder_pretrained(
                "bert-base-uncased", "bert-base-uncased"
            )
        elif self.config.base_model_name == "2layer-base":
            config_encoder = BertConfig()
            config_decoder = BertConfig()
            config_encoder.max_position_embeddings = 1090
            config_encoder.num_hidden_layers = 2
            config_decoder.num_hidden_layers = 2
            self.codec_config = EncoderDecoderConfig.from_encoder_decoder_configs(
                config_encoder, config_decoder
            )
            self.encoderdecoder = EncoderDecoderModel(config=self.codec_config)
        elif self.config.base_model_name == "3layer-base":
            config_encoder = BertConfig()
            config_decoder = BertConfig()
            config_encoder.num_hidden_layers = 3
            config_decoder.num_hidden_layers = 3
            self.codec_config = EncoderDecoderConfig.from_encoder_decoder_configs(
                config_encoder, config_decoder
            )
            self.encoderdecoder = EncoderDecoderModel(config=self.codec_config)
        if self.config.loop_contrastive:
            self.trace_caption_contrastive = TraceCaptionContrastiveModel(
                self.config.tc_contrastive_aggregate_method
            )
        if (
            hasattr(self.config, "pretrans_attention")
            and self.config.pretrans_attention
        ):

            # import ipdb; ipdb.set_trace()
            tempconf = self.encoderdecoder.config.encoder
            num_heads = tempconf.num_attention_heads
            num_layers = tempconf.num_hidden_layers
            self.attention_trans = AttentionTransform(num_layers, num_heads, 100)
        self.BOS_ID = 101
        self.vae = OpenAIDiscreteVAE()
        image_code_dim = 768
        image_fmap_size = self.vae.image_size // (2 ** self.vae.num_layers)
        self.image_seq_len = image_fmap_size ** 2
        self.image_emb = torch.nn.Embedding(self.vae.num_tokens, image_code_dim)
        self.image_pos_emb = AxialPositionalEmbedding(
            image_code_dim, axial_shape=(image_fmap_size, image_fmap_size)
        )

    def forward(self, sample_list, *args, **kwargs):

        # breakpoint()
        # import ipdb; ipdb.set_trace()
        visual_code = self.vae.get_codebook_indices(sample_list["image"])
        visual_emb = self.image_emb(visual_code)
        visual_emb += self.image_pos_emb(visual_emb)

        decoder_input_ids = sample_list["input_ids"][:, :-1]
        # using default mask
        # target_mask = sample_list["input_mask"]
        # segment_ids = sample_list["segment_ids"]
        # token_attends = sample_list["token_attends"]
        other_kwargs = {}
        # if self.config.image_feature_processor.type == "spatial":
        #     bbox_feature = sample_list["image_feature_0"]
        #     spatial_feature = sample_list["image_info_0"]["bbox"]
        #     inputs_embeds = self.image_feature_module(bbox_feature, spatial_feature)
        # else:
        #     bbox_feature = sample_list["image_feature_0"]
        #     inputs_embeds = self.image_feature_module(bbox_feature)
        # if hasattr(self.config, "no_vision") and self.config.no_vision:
        #     inputs_embeds = inputs_embeds * 0
        inputs_embeds = visual_emb
        batch_size = inputs_embeds.shape[0]
        if self.config.concate_trace:
            trace_boxes = sample_list["trace_boxes"]
            trace_boxes_mask = sample_list["trace_boxes_mask"]
            trace_feature = self.trace_feature_module(trace_boxes)
            trace_seg_id = sample_list["trace_boxes_seg_id"]
            inputs_embeds = torch.cat((inputs_embeds, trace_feature), dim=1)
            image_feats_mask = trace_boxes_mask.new_ones(
                (batch_size, visual_code.shape[1])
            )
            image_feats_seg_id = trace_seg_id.new_zeros(
                (batch_size, visual_code.shape[1])
            )
            attention_mask = torch.cat((image_feats_mask, trace_boxes_mask), dim=1)
            token_type_ids = torch.cat((image_feats_seg_id, trace_seg_id), dim=1)
            position_ids = trace_seg_id.new_zeros((batch_size, attention_mask.shape[1]))
            other_kwargs.update(
                {
                    "attention_mask": attention_mask,
                    "token_type_ids": token_type_ids,
                    "position_ids": position_ids,
                }
            )

        if self.training:
            decoder_output = self.encoderdecoder(
                decoder_input_ids=decoder_input_ids,
                inputs_embeds=inputs_embeds,
                output_attentions=True,
                output_hidden_states=True,
                return_dict=True,
                **other_kwargs
            )

            logits = decoder_output["logits"]
            cross_attentions = []
            # import ipdb; ipdb.set_trace()
            for cross_attention in decoder_output["cross_attentions"]:
                if self.config.concate_trace:
                    cross_attention = cross_attention[:, :, :, :100]
                # cross_attentions.append(cross_attention.mean(dim=1))
                cross_attentions.append(cross_attention)
            # breakpoint()
            if (
                hasattr(self.config, "pretrans_attention")
                and self.config.pretrans_attention
            ):
                cross_attentions = self.attention_trans(cross_attentions)
            else:
                cross_attentions = [crs.mean(dim=1) for crs in cross_attentions]
            model_output = {}
            model_output["captions"] = torch.max(logits, dim=-1)[1]
            model_output["scores"] = logits
            model_output["cross_attentions"] = cross_attentions
            sample_list["targets"] = sample_list["input_ids"][:, 1:]

            if self.config.loop_contrastive:
                cap_feat, vision_trace_feat = self.trace_caption_contrastive(
                    decoder_output["encoder_hidden_states"][-1],
                    sample_list["trace_boxes_loop_contrastive_seg_id"],
                    decoder_output["decoder_hidden_states"][-1],
                    sample_list["segment_ids"],
                )
                model_output["contrastive_a"] = cap_feat
                model_output["contrastive_b"] = vision_trace_feat
        else:
            if self.config.inference.type == "beam_search":
                generate_output = self.encoderdecoder.generate(
                    input_ids=None,
                    input_embeds=inputs_embeds,
                    bos_token_id=self.BOS_ID,
                    decoder_start_token_id=self.BOS_ID,
                    **self.config.inference.args,
                    **other_kwargs
                )
            elif self.config.inference.type == "greedy":
                generate_output = self.encoderdecoder.generate(
                    input_ids=None,
                    input_embeds=inputs_embeds,
                    max_length=self.config.max_gen_length,
                    bos_token_id=self.BOS_ID,
                    decoder_start_token_id=self.BOS_ID,
                    **other_kwargs
                )
            elif self.config.inference.type == "nucleus_sampling":
                generate_output = self.encoderdecoder.generate(
                    input_ids=None,
                    input_embeds=inputs_embeds,
                    bos_token_id=self.BOS_ID,
                    decoder_start_token_id=self.BOS_ID,
                    **self.config.inference.args,
                    **other_kwargs
                )
            model_output = {}
            # breakpoint()
            if (
                "return_attention" in self.config.inference
                and self.config.inference.return_attention
            ):
                with torch.no_grad():
                    attention_temp_output = self.encoderdecoder(
                        decoder_input_ids=generate_output,
                        inputs_embeds=inputs_embeds,
                        output_attentions=True,
                        return_dict=True,
                    )
                    cross_attentions = []
                    for cross_attention in attention_temp_output["cross_attentions"]:
                        if self.config.concate_trace:
                            cross_attention = cross_attention[:, :, :, :100]
                        cross_attentions.append(cross_attention.mean(dim=1))
                    # breakpoint()
                    cross_attentions = (
                        torch.stack(cross_attentions).max(dim=0)[0].max(dim=-1)[1]
                    )
                    model_output["cross_attention"] = cross_attentions
                # breakpoint()

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


class AttentionTransform(torch.nn.Module):
    def __init__(self, num_layers=2, num_heads=12, num_entries=100):
        super().__init__()
        self.linear = torch.nn.Linear(num_layers, 1)
        self.multi_head_linear = torch.nn.Linear(num_heads, 1)
        self.norm = torch.nn.LayerNorm(num_entries)

    def forward(self, attention):
        # import ipdb; ipdb.set_trace()
        attention = torch.stack(attention, dim=-1)
        attention = attention.permute(0, 2, 3, 4, 1)
        attention = self.multi_head_linear(attention).squeeze()
        attention = self.linear(attention).squeeze()
        attention = self.norm(attention)
        return attention


class TraceCaptionContrastiveModel(torch.nn.Module):
    def __init__(self, aggregate_method):
        super().__init__()
        self.vision_trace_aggregator = VisionTraceAggregator(aggregate_method)
        self.caption_aggregator = CaptionAggregator(aggregate_method)

    def forward(self, vision_trace_feat, vision_trace_mask, caption_feat, caption_mask):

        # import ipdb; ipdb.set_trace()
        # caption information aggregate
        # generate a feat list [bs, Tensor(num_sentences, feat)]
        caption_feats = self.caption_aggregator(caption_feat, caption_mask)

        # vision & trace infomation aggregate
        # generate a feat list [bs, Tensor(num_trace_segment, feat)]
        vision_trace_feats = self.vision_trace_aggregator(
            vision_trace_feat, vision_trace_mask
        )

        # in batch permutation?
        # move to loss part

        return caption_feats, vision_trace_feats


class VisionTraceAggregator(torch.nn.Module):
    def __init__(self, aggregate_method, hidden_size=768):
        super().__init__()
        self.aggregate_method = aggregate_method
        if aggregate_method == "maxpool":
            self.aggregator = lambda x: torch.max(x, dim=1)
        elif aggregate_method == "meanpool":
            self.aggregator = lambda x: torch.mean(x, dim=0)
        elif aggregate_method == "lstm":
            self.aggregator = torch.nn.LSTM(
                hidden_size, hidden_size, 2, bidirectional=True
            )
        elif aggregate_method == "transformer":
            encoder_layer = torch.nn.TransformerEncoderLayer(
                d_model=hidden_size, nhead=8
            )
            self.aggregator = torch.nn.TransformerEncoder(
                encoder_layer=encoder_layer, num_layers=2
            )

        self.vt_merge = torch.nn.Linear(2 * hidden_size, hidden_size)

    def forward(self, vision_trace_feat, vision_trace_mask):
        vision_feat = vision_trace_feat[:, :100].mean(axis=1)
        trace_feat = vision_trace_feat[:, 100:]
        trace_mask = vision_trace_mask
        max_seg_id = trace_mask.max().item()
        feat_list = []
        section_sizes = []
        # import ipdb; ipdb.set_trace()
        for seg_id in range(1, max_seg_id + 1):
            mask = trace_mask == seg_id
            section_sizes.append(mask.sum(axis=1))
        # [bs, max_seg_id] the element is the seq_length of segment
        # with seg_id in current instance
        section_sizes = torch.stack(section_sizes).t()
        batch_section_count = (section_sizes > 0).sum(axis=1)
        # [bs * num(seglen > 0)]
        section_sizes_flatten_wo_0 = section_sizes[section_sizes > 0]

        trace_feat = trace_feat[trace_mask > 0]

        # assert caption_feat.shape[0] == sum(section_sizes)
        # (num_total_sentences, Tensor(sentence_len, feat_dim))
        debatched_trace_feat = torch.split(
            trace_feat, section_sizes_flatten_wo_0.tolist()
        )
        if self.aggregate_method == "maxpool":
            # [num_total_sentences, sentence_max_len, feat_dim]
            debatched_trace_feat = torch.nn.utils.rnn.pad_sequence(
                debatched_trace_feat, batch_first=True
            )
            feat_aggs, _ = self.aggregator(debatched_trace_feat)
        elif self.aggregate_method == "meanpool":
            feat_aggs = []
            for feat in debatched_trace_feat:
                feat_agg = self.aggregator(feat)
                feat_aggs.append(feat_agg)
            feat_aggs = torch.stack(feat_aggs)
        elif self.aggregate_method == "lstm":
            debatched_trace_feat = torch.nn.utils.rnn.pad_sequence(debatched_trace_feat)
            packed_trace_feat = torch.nn.utils.rnn.pack_padded_sequence(
                debatched_trace_feat,
                section_sizes_flatten_wo_0.tolist(),
                enforce_sorted=False,
            )
            output, (h_n, c_n) = self.aggregator(packed_trace_feat)
            h_n = h_n.view(2, 2, section_sizes_flatten_wo_0.shape[0], 768)
            feat_aggs = h_n[-1].squeeze(0).mean(0)
        elif self.aggregate_method == "transformer":
            debatched_trace_feat = torch.nn.utils.rnn.pad_sequence(debatched_trace_feat)
            max_seg_len, batch, _ = debatched_trace_feat.shape
            mask = torch.arange(
                0, max_seg_len, dtype=torch.long, device=trace_feat.device
            ).repeat(batch)
            mask = (
                mask < section_sizes_flatten_wo_0.repeat_interleave(max_seg_len)
            ).view(batch, max_seg_len)
            mask = ~mask
            # padding mask not working
            h = self.aggregator(debatched_trace_feat, None, mask).transpose(0, 1)
            feat_aggs = h.mean(axis=1)
        vision_feat_expand = []
        for v_, size in zip(vision_feat, batch_section_count.tolist()):
            vision_feat_expand.append(v_.repeat(size, 1))
        vision_feat_expand = torch.cat(vision_feat_expand)

        # import ipdb; ipdb.set_trace()
        vt_feat_aggs = torch.cat([feat_aggs, vision_feat_expand], dim=1)
        vt_feat_aggs = self.vt_merge(vt_feat_aggs)

        feat_list = torch.split(vt_feat_aggs, batch_section_count.tolist())

        return feat_list


class CaptionAggregator(torch.nn.Module):
    def __init__(self, aggregate_method, hidden_size=768):
        super().__init__()
        self.aggregate_method = aggregate_method
        if aggregate_method == "maxpool":
            self.aggregator = lambda x: torch.max(x, dim=1)
        elif aggregate_method == "meanpool":
            self.aggregator = lambda x: torch.mean(x, dim=0)
        elif aggregate_method == "lstm":
            self.aggregator = torch.nn.LSTM(
                hidden_size, hidden_size, 2, bidirectional=True
            )
        elif aggregate_method == "transformer":
            encoder_layer = torch.nn.TransformerEncoderLayer(
                d_model=hidden_size, nhead=8
            )
            self.aggregator = torch.nn.TransformerEncoder(
                encoder_layer=encoder_layer, num_layers=2
            )

    def forward(self, caption_feat, caption_mask):
        # remove bos
        # import ipdb; ipdb.set_trace()
        caption_mask = caption_mask[:, 1:]
        max_seg_id = caption_mask.max().item()
        feat_list = []
        section_sizes = []
        for seg_id in range(1, max_seg_id + 1):
            mask = caption_mask == seg_id
            section_sizes.append(mask.sum(axis=1))
        # [bs, max_seg_id] the element is the seq_length of segment
        # with seg_id in current instance
        section_sizes = torch.stack(section_sizes).t()
        batch_section_count = (section_sizes > 0).sum(axis=1)
        # [bs * num(seglen > 0)]
        section_sizes_flatten_wo_0 = section_sizes[section_sizes > 0]

        caption_feat = caption_feat[caption_mask > 0]

        # assert caption_feat.shape[0] == sum(section_sizes)
        # (num_total_sentences, Tensor(sentence_len, feat_dim))
        debatched_caption_feat = torch.split(
            caption_feat, section_sizes_flatten_wo_0.tolist()
        )
        if self.aggregate_method == "maxpool":
            # [num_total_sentences, sentence_max_len, feat_dim]
            debatched_caption_feat = torch.nn.utils.rnn.pad_sequence(
                debatched_caption_feat, batch_first=True
            )
            feat_aggs, _ = self.aggregator(debatched_caption_feat)
        elif self.aggregate_method == "meanpool":
            feat_aggs = []
            for feat in debatched_caption_feat:
                feat_agg = self.aggregator(feat)
                feat_aggs.append(feat_agg)
            feat_aggs = torch.stack(feat_aggs)
        elif self.aggregate_method == "lstm":
            debatched_caption_feat = torch.nn.utils.rnn.pad_sequence(
                debatched_caption_feat
            )
            packed_caption_feat = torch.nn.utils.rnn.pack_padded_sequence(
                debatched_caption_feat,
                section_sizes_flatten_wo_0.tolist(),
                enforce_sorted=False,
            )
            output, (h_n, c_n) = self.aggregator(packed_caption_feat)
            h_n = h_n.view(2, 2, section_sizes_flatten_wo_0.shape[0], 768)
            feat_aggs = h_n[-1].squeeze(0).mean(0)
        elif self.aggregate_method == "transformer":
            debatched_caption_feat = torch.nn.utils.rnn.pad_sequence(
                debatched_caption_feat
            )
            max_sentence_len, batch, _ = debatched_caption_feat.shape
            mask = torch.arange(
                0, max_sentence_len, dtype=torch.long, device=caption_feat.device
            ).repeat(batch)
            mask = (
                mask < section_sizes_flatten_wo_0.repeat_interleave(max_sentence_len)
            ).view(batch, max_sentence_len)
            mask = ~mask
            # padding mask not working
            h = self.aggregator(debatched_caption_feat, None, mask).transpose(0, 1)
            feat_aggs = h.mean(axis=1)
        # import ipdb; ipdb.set_trace()

        feat_list = torch.split(feat_aggs, batch_section_count.tolist())
        return feat_list
