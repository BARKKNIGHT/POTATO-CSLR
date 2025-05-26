class ViViT_SLR(nn.Module):
    def __init__(self,
                 vivit_config=VivitConfig(),
                 vocab_size=len(idx_to_word),
                 d_model=768,
                 nhead=8,
                 num_decoder_layers=4,
                 dim_feedforward=3072,
                 dropout=0.1,
                 batch_first=True,
                 num_heads=4,
                 pad_token=0,
                 sos_token=1,
                 eos_token=2,
                 residual_ratio=0,
                 max_pred=512):

        super(ViViT_SLR, self).__init__()

        # self.vivit = VivitModel(vivit_config)
        self.vivit = VivitModel.from_pretrained("google/vivit-b-16x2-kinetics400", attn_implementation="sdpa", torch_dtype=torch.float32)
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

        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=batch_first)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self.pad_token = pad_token
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token)
        self.max_pred = max_pred

        self.fc_out = nn.Linear(d_model, vocab_size)

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


        if y_tokens.shape[-1] == 2:
            # all_logits = torch.zeros(self.vocab_size).unsqueeze(0).unsqueeze(0).to(device)

            for i in range(self.max_pred-2):
                y_embedded = self.embedding(y_tokens)
                tgt_seq_len = y_embedded.shape[1]
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_seq_len).to(y_embedded.device)

                tgt_key_padding_mask = (y_tokens == self.pad_token)

                decoded_output = self.decoder(y_embedded, attn_output, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
                logit = self.fc_out(decoded_output)
                # all_logits = torch.cat([all_logits, logit[:,-1,:].unsqueeze(0)], dim=-2)


                last_output_token = logit[:,-1,:].argmax(1).unsqueeze(0)
                y_tokens = torch.cat([y_tokens, last_output_token], dim=1)
                # y_tokens = logit[:,-1,:].argmax(1).unsqueeze(0)
                print(y_tokens)

                if (last_output_token == self.eos_token).all():
                    return y_tokens, logit

            else:
                y_tokens = torch.cat([y_tokens, torch.tensor([[self.eos_token]]).to(device)], dim=1)
                return y_tokens, logit

        else:
            y_embedded = self.embedding(y_tokens)

            tgt_seq_len = y_embedded.shape[1]
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_seq_len).to(y_embedded.device)

            tgt_key_padding_mask = (y_tokens == self.pad_token)

            decoded_output = self.decoder(y_embedded, attn_output, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)

            logit = self.fc_out(decoded_output)

            return logit

    def init_weights(self):
        # Embedding
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        if self.embedding.padding_idx is not None:
            with torch.no_grad():
                self.embedding.weight[self.embedding.padding_idx].fill_(0)

        # Output Linear
        nn.init.xavier_uniform_(self.fc_out.weight)
        if self.fc_out.bias is not None:
            nn.init.constant_(self.fc_out.bias, 0)

        # Decoder layers
        for layer in self.decoder.layers:
            # Self-attention
            nn.init.xavier_uniform_(layer.self_attn.in_proj_weight)
            if layer.self_attn.in_proj_bias is not None:
                nn.init.constant_(layer.self_attn.in_proj_bias, 0)
            # Cross-attention
            nn.init.xavier_uniform_(layer.multihead_attn.in_proj_weight)
            if layer.multihead_attn.in_proj_bias is not None:
                nn.init.constant_(layer.multihead_attn.in_proj_bias, 0)
            # Linear layers
            nn.init.xavier_uniform_(layer.linear1.weight)
            nn.init.xavier_uniform_(layer.linear2.weight)
            if layer.linear1.bias is not None:
                nn.init.constant_(layer.linear1.bias, 0)
            if layer.linear2.bias is not None:
                nn.init.constant_(layer.linear2.bias, 0)
            # LayerNorm
            nn.init.constant_(layer.norm1.weight, 1)
            nn.init.constant_(layer.norm1.bias, 0)
            nn.init.constant_(layer.norm2.weight, 1)
            nn.init.constant_(layer.norm2.bias, 0)
            if hasattr(layer, 'norm3'):
                nn.init.constant_(layer.norm3.weight, 1)
                nn.init.constant_(layer.norm3.bias, 0)
