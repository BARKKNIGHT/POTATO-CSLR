class ViViT_SLR(nn.Module):
    def __init__(self,
                 vivit_weights = "google/vivit-b-16x2-kinetics400",
                 t5_weights = "google-t5/t5-base",
                 vocab_size = 1299,
                 batch_first=True,
                 num_heads=4,
                 pad_token=0,
                 sos_token=1,
                 eos_token=2,
                 residual_ratio=0,
                 max_pred=512):

        super(ViViT_SLR, self).__init__()

        self.vivit_weights = vivit_weights
        self.t5_weights = t5_weights
        self.vivit = VivitModel.from_pretrained(self.vivit_weights, attn_implementation="sdpa", torch_dtype=torch.float32)
        r3d_18 = torchvision.models.video.r3d_18(pretrained=True)
        self.r3d_18 = nn.Sequential(
            r3d_18.stem,
            r3d_18.layer1,
            r3d_18.layer2,
            r3d_18.layer3,
            nn.Conv3d(256, 512, kernel_size=(2, 1, 1), stride=(2, 1, 1))
        )
        self.cross_attn = nn.MultiheadAttention(embed_dim=768, num_heads=num_heads, batch_first=batch_first)

        self.linear = nn.Sequential(
            nn.Linear(in_features=512, out_features=640),
            nn.ReLU(),
            nn.Linear(in_features=640, out_features=768)
        )

        self.non_linear = nn.Sequential(
            nn.Linear(in_features=512, out_features=640),
            nn.ReLU(),
            nn.Linear(in_features=640, out_features=768)
        )
        self.linear = nn.Linear(512, 768)
        
        self.residual_ratio = residual_ratio
        
        self.normalize = nn.LayerNorm(768)

        self.t5 = self.copy_pretrained_T5_weights()

        self.pad_token = pad_token
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.vocab_size = vocab_size

        self.max_pred = max_pred

        self.fc_out = nn.Linear(768, vocab_size)

        self.init_weights()


    def forward(self, img, y_tokens):

        
        encoded_output1 = self.vivit(img).last_hidden_state
        encoded_output2 = self.r3d_18(img.permute(0,2,1,3,4))
        encoded_output2 = encoded_output2.permute(0,2,3,4,1)
        encoded_output2 = encoded_output2.view(-1, 28*28*4,512)
        encoded_output2 = self.residual_ratio * self.linear(encoded_output2) + (1 - self.residual_ratio) * self.non_linear(encoded_output2)
        encoded_output1 = self.normalize(encoded_output1)
        encoded_output2 = self.normalize(encoded_output2)

        attn_output, _ = self.cross_attn(query=encoded_output1[:, :3136, :], key=encoded_output2[:, :3136, :], value=encoded_output1[:, :3136, :])

        # print("Encoded Output 1 (ViViT)")
        # print("Min:", encoded_output1.min().item())
        # print("Max:", encoded_output1.max().item())
        # print("Mean:", encoded_output1.mean().item())
        # print("Std Dev:", encoded_output1.std().item())
        # print("-" * 30)

        # print("Encoded Output 2 (r3d_18)")
        # print("Min:", encoded_output2.min().item())
        # print("Max:", encoded_output2.max().item())
        # print("Mean:", encoded_output2.mean().item())
        # print("Std Dev:", encoded_output2.std().item())
        # print("-" * 30)

        # print("Cross Attention Output")
        # print("Min:", attn_output.min().item())
        # print("Max:", attn_output.max().item())
        # print("Mean:", attn_output.mean().item())
        # print("Std Dev:", attn_output.std().item())
        # print("-" * 30)

        decoder_attention_mask = (y_tokens != self.pad_token).long()
        decoded_output = self.t5.decoder(input_ids=y_tokens, encoder_hidden_states=attn_output, attention_mask=decoder_attention_mask).last_hidden_state
        logit = self.fc_out(decoded_output)

        return logit

    def init_weights(self):

        nn.init.xavier_uniform_(self.fc_out.weight)
        if self.fc_out.bias is not None:
            nn.init.constant_(self.fc_out.bias, 0)

    def copy_pretrained_T5_weights(self):

        config = T5Config.from_pretrained(self.t5_weights)
        config.vocab_size = self.vocab_size
        config.decoder_start_token_id = self.sos_token9
        new_model = T5ForConditionalGeneration(config)
        pretrained_model = T5ForConditionalGeneration.from_pretrained(self.t5_weights)

        pretrained_dict = pretrained_model.state_dict()
        new_dict = new_model.state_dict()

        filtered_dict = {}
        for k, v in pretrained_dict.items():
            if (
                k in new_dict
                and not k.startswith("shared")
                and "embed_tokens" not in k
                and "lm_head" not in k
                and new_dict[k].shape == v.shape 
            ):
                filtered_dict[k] = v

        print(f"Copying {len(filtered_dict)} weights from pretrained T5 model.")
        new_dict.update(filtered_dict)
        new_model.load_state_dict(new_dict)
        return new_model
