class ViViT_SLR(nn.Module):
    def __init__(self,
                 vivit_weights = "google/vivit-b-16x2-kinetics400",
                 t5_weights = "google-t5/t5-base",
                 vocab_size = len(idx_to_word),
                 batch_first=True,
                 num_heads=12,
                 dropout=0.1,
                 pad_token=0,
                 sos_token=1,
                 eos_token=2,
                 max_pred=512,
                 early_attention=3,
                 mid_attention=7):

        super(ViViT_SLR, self).__init__()

        self.vivit_weights = vivit_weights
        self.t5_weights = t5_weights
        self.vivit_normal = VivitModel.from_pretrained(self.vivit_weights, attn_implementation="sdpa", torch_dtype=torch.float32, use_safetensors=True)
        self.vivit_keypoint = VivitModel.from_pretrained(self.vivit_weights, attn_implementation="sdpa", torch_dtype=torch.float32, use_safetensors=True)


        self.early_cross_attn_normal_to_kp = nn.MultiheadAttention(embed_dim=768, num_heads=num_heads, dropout=dropout, batch_first=batch_first)
        self.early_cross_attn_kp_to_normal = nn.MultiheadAttention(embed_dim=768, num_heads=num_heads, dropout=dropout, batch_first=batch_first)
        self.mid_cross_attn_normal_to_kp  = nn.MultiheadAttention(embed_dim=768, num_heads=num_heads, dropout=dropout, batch_first=batch_first)
        self.mid_cross_attn_kp_to_normal = nn.MultiheadAttention(embed_dim=768, num_heads=num_heads, dropout=dropout, batch_first=batch_first)

        self.cross_attn_normal_to_kp = nn.MultiheadAttention(embed_dim=768, num_heads=num_heads, dropout=dropout, batch_first=batch_first)
        self.cross_attn_kp_to_normal = nn.MultiheadAttention(embed_dim=768, num_heads=num_heads, dropout=dropout, batch_first=batch_first)

        self.cross_attn_decoder = nn.MultiheadAttention(embed_dim=768, num_heads=num_heads, dropout=dropout, batch_first=batch_first)


        self.early_norm_kp_to_normal = nn.LayerNorm(768, eps=1e-6)
        self.early_norm_normal_to_kp = nn.LayerNorm(768, eps=1e-6)

        self.mid_norm_kp_to_normal = nn.LayerNorm(768, eps=1e-6)
        self.mid_norm_normal_to_kp = nn.LayerNorm(768, eps=1e-6)

        self.norm_normal = nn.LayerNorm(768, eps=1e-6)
        self.norm_keypoint = nn.LayerNorm(768, eps=1e-6)
        self.norm_normal_to_kp = nn.LayerNorm(768, eps=1e-6)
        self.norm_kp_to_normal = nn.LayerNorm(768, eps=1e-6)

        self.vocab_size = vocab_size
        self.pad_token = pad_token
        self.sos_token = sos_token
        self.eos_token = eos_token

        self.t5_normal = self.copy_pretrained_T5_weights()
        self.t5_keypoint = self.copy_pretrained_T5_weights()
        self.freeze_unused_tokens()

        self.shared_embedding = self.t5_keypoint.shared
        self.t5_normal.decoder.embed_tokens = self.shared_embedding
        self.t5_keypoint.decoder.embed_tokens = self.shared_embedding

        self.lm_head = self.t5_normal.lm_head

        self.max_pred = max_pred
        self.early_attention = early_attention
        self.mid_attention = mid_attention


    def forward(self, normal_vid, keypoint_vid, y_tokens):

        normal_encoded_output = self.vivit_normal.embeddings(normal_vid)
        keypoint_encoded_output = self.vivit_keypoint.embeddings(keypoint_vid)

        for i in range(12):
            normal_encoded_output = self.vivit_normal.encoder.layer[i](normal_encoded_output)[0]
            keypoint_encoded_output = self.vivit_keypoint.encoder.layer[i](keypoint_encoded_output)[0]

            if i == self.early_attention:
                early_normal_to_kp, _ = self.early_cross_attn_normal_to_kp(query=normal_encoded_output, key=keypoint_encoded_output, value=keypoint_encoded_output)
                early_kp_to_normal, _ = self.early_cross_attn_kp_to_normal(query=keypoint_encoded_output, key=normal_encoded_output, value=normal_encoded_output)

                normal_encoded_output = self.early_norm_normal_to_kp(normal_encoded_output + early_normal_to_kp)
                keypoint_encoded_output = self.early_norm_kp_to_normal(keypoint_encoded_output + early_kp_to_normal)

            elif i == self.mid_attention:
                mid_normal_to_kp, _ = self.mid_cross_attn_normal_to_kp(query=normal_encoded_output, key=keypoint_encoded_output, value=keypoint_encoded_output)
                mid_kp_to_normal, _ = self.mid_cross_attn_kp_to_normal(query=keypoint_encoded_output, key=normal_encoded_output, value=normal_encoded_output)

                normal_encoded_output = self.mid_norm_normal_to_kp(normal_encoded_output + mid_normal_to_kp)
                keypoint_encoded_output = self.mid_norm_kp_to_normal(keypoint_encoded_output + mid_kp_to_normal)

        normal_encoded_output = self.norm_normal(normal_encoded_output)
        keypoint_encoded_output = self.norm_keypoint(keypoint_encoded_output)

        normal_to_kp, _ = self.cross_attn_normal_to_kp(query=normal_encoded_output, key=keypoint_encoded_output, value=keypoint_encoded_output)
        kp_to_normal, _ = self.cross_attn_kp_to_normal(query=keypoint_encoded_output, key=normal_encoded_output, value=normal_encoded_output)

        normal_encoded_output = self.norm_normal_to_kp(normal_encoded_output + normal_to_kp)
        keypoint_encoded_output = self.norm_kp_to_normal(keypoint_encoded_output + kp_to_normal)


        decoder_attention_mask = (y_tokens != self.pad_token).long()

        decoder_input_embeds = self.shared_embedding(y_tokens)

        normal_decoded_output = self.t5_normal.decoder(inputs_embeds=decoder_input_embeds, encoder_hidden_states=normal_encoded_output, attention_mask=decoder_attention_mask).last_hidden_state
        keypoint_decoded_output = self.t5_keypoint.decoder(inputs_embeds=decoder_input_embeds, encoder_hidden_states=keypoint_encoded_output, attention_mask=decoder_attention_mask).last_hidden_state

        attn_output, _ = self.cross_attn_decoder(query=keypoint_decoded_output, key=normal_decoded_output, value=normal_decoded_output)

        logits = self.lm_head(attn_output)
        return logits[:, :, :self.vocab_size]


    def copy_pretrained_T5_weights(self):
        config = T5Config.from_pretrained(self.t5_weights, use_safetensors=True)
        config.vocab_size = self.vocab_size
        config.decoder_start_token_id = self.sos_token
        config.eos_token_id = self.eos_token

        new_model = T5ForConditionalGeneration(config)
        pretrained_model = T5ForConditionalGeneration.from_pretrained(self.t5_weights, use_safetensors=True)

        # Manually assign shared embeddings, decoder embeddings, and lm_head
        new_model.shared = pretrained_model.shared
        new_model.decoder.embed_tokens = pretrained_model.decoder.embed_tokens
        new_model.lm_head = pretrained_model.lm_head

        # Copy other matching weights
        pretrained_dict = pretrained_model.state_dict()
        new_dict = new_model.state_dict()

        filtered_dict = {}
        for k, v in pretrained_dict.items():
            if k in new_dict and new_dict[k].shape == v.shape:
                filtered_dict[k] = v

        print(f"Copying {len(filtered_dict)} weights from pretrained T5 model.")
        new_dict.update(filtered_dict)
        new_model.load_state_dict(new_dict)

        return new_model


    def freeze_unused_tokens(self):
        for model in [self.t5_normal, self.t5_keypoint]:
            allowed_ids = self.vocab_size
            embedding = model.shared.weight
            lm_head = model.lm_head.weight

            all_ids = torch.arange(embedding.shape[0], device=embedding.device)
            frozen_mask = ~(all_ids < allowed_ids)

            with torch.no_grad():
                embedding[frozen_mask] = embedding[frozen_mask].detach()
                lm_head[frozen_mask] = lm_head[frozen_mask].detach()

            def _hook(grad):
                grad[frozen_mask] = 0
                return grad

            embedding.register_hook(_hook)
            lm_head.register_hook(_hook)
