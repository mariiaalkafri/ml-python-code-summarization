import torch
import torch.nn as nn
import torch.nn.functional as F


class LuongAttention(nn.Module):
    def __init__(self, hidden_size):
        super(LuongAttention, self).__init__()
        self.hidden_size = hidden_size
        self.project = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden, encoder_outputs, src_mask=None):
        # hidden: [B, 1, H]
        # encoder_outputs: [B, S, H]

        if hidden.dim() == 2:
            hidden = hidden.unsqueeze(1)

        proj_enc = self.project(encoder_outputs)  # [B, S, H]
        attn_scores = torch.bmm(hidden, proj_enc.transpose(1, 2))  # [B, 1, S]

        if src_mask is not None:
            mask = src_mask.unsqueeze(1)  # [B,1,S]
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(attn_scores, dim=-1)  # [B,1,S]
        context = torch.bmm(attn_weights, encoder_outputs)  # [B,1,H]
        return context, attn_weights


class Seq2SeqLSTMAttn(nn.Module):
    def __init__(
        self,
        vocab_size,
        emb_dim=256,
        enc_hidden=256,
        dec_hidden=512,
        num_layers=1,
        dropout=0.2,
        pad_id=0
    ):
        super(Seq2SeqLSTMAttn, self).__init__()

        self.pad_id = pad_id
        self.enc_out_dim = enc_hidden * 2

        self.src_embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.tgt_embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)

        self.encoder = nn.LSTM(
            input_size=emb_dim,
            hidden_size=enc_hidden,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.decoder = nn.LSTM(
            input_size=emb_dim + self.enc_out_dim,
            hidden_size=dec_hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # IMPORTANT: keep this name to match checkpoint keys:
        self.attention = LuongAttention(dec_hidden)

        self.h_bridge = nn.Linear(self.enc_out_dim, dec_hidden)
        self.c_bridge = nn.Linear(self.enc_out_dim, dec_hidden)

        self.dropout = nn.Dropout(dropout)
        self.concat = nn.Linear(dec_hidden + self.enc_out_dim, dec_hidden)
        self.out = nn.Linear(dec_hidden, vocab_size)

    def forward(self, src_ids, src_mask, tgt_in):
        batch_size = src_ids.size(0)

        src_emb = self.dropout(self.src_embedding(src_ids))
        enc_output, (enc_h, enc_c) = self.encoder(src_emb)

        h_cnc = torch.cat([enc_h[-2], enc_h[-1]], dim=1)
        c_cnc = torch.cat([enc_c[-2], enc_c[-1]], dim=1)

        dec_h = torch.tanh(self.h_bridge(h_cnc)).unsqueeze(0)
        dec_c = torch.tanh(self.c_bridge(c_cnc)).unsqueeze(0)

        decoder_hidden = (dec_h, dec_c)

        tgt_emb = self.dropout(self.tgt_embedding(tgt_in))

        seq_len = tgt_in.size(1)
        outputs = []

        context = torch.zeros(batch_size, 1, self.enc_out_dim, device=src_ids.device)

        for t in range(seq_len):
            input_t = tgt_emb[:, t:t+1, :]
            rnn_input = torch.cat([input_t, context], dim=2)

            dec_out, decoder_hidden = self.decoder(rnn_input, decoder_hidden)

            # NOTE: this is as in your training setup (keep it to load checkpoint)
            # Even if imperfect dimensionally, we keep it for compatibility.
            context, _ = self.attention(dec_out, enc_output, src_mask)

            concat_input = torch.cat([dec_out, context], dim=2)
            concat_out = torch.tanh(self.concat(concat_input))

            logits = self.out(concat_out)
            outputs.append(logits)

        outputs = torch.cat(outputs, dim=1)
        return outputs

    @staticmethod
    def _get_ngrams(tokens, n: int):
        if len(tokens) < n:
            return set()
        return set(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))

    def generate(
        self,
        src_ids,
        src_mask,
        max_len=50,
        bos_id=None,
        eos_id=None,
        no_repeat_ngram_size=3,
        repetition_penalty=1.15
    ):
        """
        Fixed generation:
        - Works for batch size > 1
        - Stops per-sample at EOS
        - Adds repetition controls to avoid "b b b b ..."
        Returns: List[List[int]] (each includes BOS then generated tokens)
        """
        if bos_id is None or eos_id is None:
            raise ValueError("bos_id and eos_id must be provided for generation")

        self.eval()
        device = src_ids.device
        B = src_ids.size(0)

        # Encode
        src_emb = self.dropout(self.src_embedding(src_ids))
        enc_output, (enc_h, enc_c) = self.encoder(src_emb)

        # Init decoder state
        h_cnc = torch.cat([enc_h[-2], enc_h[-1]], dim=1)
        c_cnc = torch.cat([enc_c[-2], enc_c[-1]], dim=1)

        dec_h = torch.tanh(self.h_bridge(h_cnc)).unsqueeze(0)
        dec_c = torch.tanh(self.c_bridge(c_cnc)).unsqueeze(0)

        decoder_hidden = (dec_h, dec_c)

        # Start tokens
        ys = torch.full((B, 1), bos_id, dtype=torch.long, device=device)  # [B,1]
        context = torch.zeros(B, 1, self.enc_out_dim, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_len):
            decoder_input = ys[:, -1:]  # last token
            tgt_emb = self.dropout(self.tgt_embedding(decoder_input))  # [B,1,E]

            rnn_input = torch.cat([tgt_emb, context], dim=2)
            dec_out, decoder_hidden = self.decoder(rnn_input, decoder_hidden)

            context, _ = self.attention(dec_out, enc_output, src_mask)

            concat_input = torch.cat([dec_out, context], dim=2)
            concat_out = torch.tanh(self.concat(concat_input))

            logits = self.out(concat_out).squeeze(1)  # [B,V]

            # repetition penalty
            if repetition_penalty is not None and repetition_penalty > 1.0:
                for b in range(B):
                    used = ys[b].tolist()
                    logits[b, used] = logits[b, used] / repetition_penalty

            # no-repeat ngram blocking
            if no_repeat_ngram_size is not None and no_repeat_ngram_size > 1:
                n = no_repeat_ngram_size
                for b in range(B):
                    if finished[b]:
                        continue
                    prev = ys[b].tolist()
                    if len(prev) >= n - 1:
                        prefix = tuple(prev[-(n - 1):])
                        ngrams = self._get_ngrams(prev, n)
                        banned = []
                        for cand in range(logits.size(-1)):
                            if prefix + (cand,) in ngrams:
                                banned.append(cand)
                        if banned:
                            logits[b, banned] = -1e9

            next_token = torch.argmax(logits, dim=-1)  # [B]
            next_token = torch.where(
                finished,
                torch.tensor(eos_id, device=device),
                next_token
            )

            ys = torch.cat([ys, next_token.unsqueeze(1)], dim=1)
            finished = finished | (next_token == eos_id)

            if torch.all(finished):
                break

        return [seq.tolist() for seq in ys]
