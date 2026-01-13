import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: [B, T, E]
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerSeq2Seq(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=256,
        nhead=8,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dim_feedforward=1024,
        dropout=0.1,
        pad_id=0
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_id = pad_id

        self.src_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.tgt_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos = PositionalEncoding(d_model, dropout)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,   # ✅ makes life easy
            norm_first=True
        )

        self.out = nn.Linear(d_model, vocab_size)

    def _causal_mask(self, T, device):
        # True = block
        return torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)

    def forward(self, src_ids, src_mask, tgt_in):
        """
        Matches your LSTM interface:
          src_ids:  [B,S]
          src_mask: [B,S]  (1=real, 0=pad)
          tgt_in:   [B,T]
        returns logits: [B,T,V]
        """
        device = src_ids.device

        src = self.src_embedding(src_ids) * math.sqrt(self.d_model)  # [B,S,E]
        tgt = self.tgt_embedding(tgt_in) * math.sqrt(self.d_model)   # [B,T,E]
        src = self.pos(src)
        tgt = self.pos(tgt)

        src_key_padding_mask = (src_mask == 0)         # [B,S] True=pad
        tgt_key_padding_mask = (tgt_in == self.pad_id) # [B,T] True=pad

        T = tgt_in.size(1)
        tgt_causal_mask = self._causal_mask(T, device=device)  # [T,T]

        out = self.transformer(
            src=src,
            tgt=tgt,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
            tgt_mask=tgt_causal_mask,
        )  # [B,T,E]

        logits = self.out(out)  # [B,T,V]
        return logits

    @torch.no_grad()
    def generate(self, src_ids, src_mask, max_len=50, bos_id=None, eos_id=None):
        """
        Greedy generation. Returns List[List[int]] each starting with BOS.
        """
        if bos_id is None or eos_id is None:
            raise ValueError("bos_id and eos_id must be provided")

        self.eval()
        device = src_ids.device
        B = src_ids.size(0)

        ys = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_len):
            logits = self.forward(src_ids, src_mask, ys)     # [B,t,V]
            next_token = logits[:, -1].argmax(dim=-1)        # [B]

            next_token = torch.where(
                finished,
                torch.tensor(eos_id, device=device),
                next_token
            )

            ys = torch.cat([ys, next_token.unsqueeze(1)], dim=1)
            finished |= (next_token == eos_id)
            if torch.all(finished):
                break

        return [seq.tolist() for seq in ys]
