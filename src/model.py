import torch
import torch.nn as nn
import torch.nn.functional as F


class LuongAttention(nn.Module):
    """
    Luong (multiplicative) attention with a projection on encoder outputs.
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.project = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden, encoder_outputs, src_mask=None):
        """
        hidden: [B, 1, H]  (decoder output for current step)
        encoder_outputs: [B, S, H]  (encoder hidden states)
        src_mask: [B, S] (1 for real tokens, 0 for padding)
        """
        if hidden.dim() == 2:
            hidden = hidden.unsqueeze(1)  # [B, 1, H]

        proj_enc = self.project(encoder_outputs)  # [B, S, H]
        attn_scores = torch.bmm(hidden, proj_enc.transpose(1, 2))  # [B, 1, S]

        if src_mask is not None:
            # src_mask: [B, S] -> [B, 1, S]
            mask = src_mask.unsqueeze(1)
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, 1, S]
        context = torch.bmm(attn_weights, encoder_outputs)  # [B, 1, H]
        return context, attn_weights


class Seq2SeqLSTMAttn(nn.Module):
    """
    BiLSTM encoder + LSTM decoder with Luong attention.
    Training forward uses teacher forcing (tgt_in given).
    Generation uses greedy decoding with anti-repetition constraints.
    """
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int = 256,
        enc_hidden: int = 256,
        dec_hidden: int = 512,
        num_layers: int = 1,
        dropout: float = 0.2,
        pad_id: int = 0
    ):
        super().__init__()

        self.pad_id = pad_id
        self.enc_out_dim = enc_hidden * 2  # because bidirectional encoder

        self.src_embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.tgt_embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)

        self.encoder = nn.LSTM(
            input_size=emb_dim,
            hidden_size=enc_hidden,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Decoder input: target embedding + attention context
        self.decoder = nn.LSTM(
            input_size=emb_dim + self.enc_out_dim,
            hidden_size=dec_hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Attention works on decoder hidden, but encoder outputs are enc_out_dim.
        # Your original attention projected encoder outputs using dec_hidden size,
        # which is dimension-mismatched. However you were passing dec_out [B,1,dec_hidden]
        # and enc_output [B,S,enc_out_dim]. That only works if project expects enc_out_dim,
        # not dec_hidden.
        #
        # FIX: Use LuongAttention with hidden_size = enc_out_dim? No. Luong attention compares
        # decoder hidden with projected encoder outputs => they must match dimensions.
        #
        # EASIEST FIX (keeping your architecture): project encoder outputs to dec_hidden.
        # So attention module should project enc_out_dim -> dec_hidden, then dot with dec_hidden.
        self.attn_proj = nn.Linear(self.enc_out_dim, dec_hidden, bias=False)

        self.h_bridge = nn.Linear(self.enc_out_dim, dec_hidden)
        self.c_bridge = nn.Linear(self.enc_out_dim, dec_hidden)

        self.dropout = nn.Dropout(dropout)

        # Combine decoder hidden + (original) encoder context
        self.concat = nn.Linear(dec_hidden + self.enc_out_dim, dec_hidden)
        self.out = nn.Linear(dec_hidden, vocab_size)

    def _encode(self, src_ids: torch.Tensor):
        """
        Returns:
          enc_output: [B, S, enc_out_dim]
          decoder_hidden: (h, c) each [num_layers, B, dec_hidden]
        """
        src_emb = self.dropout(self.src_embedding(src_ids))  # [B, S, E]
        enc_output, (enc_h, enc_c) = self.encoder(src_emb)   # enc_output [B,S,2*enc_hidden]

        # Take last layer's forward/backward hidden states
        # enc_h: [num_layers*2, B, enc_hidden]
        h_cnc = torch.cat([enc_h[-2], enc_h[-1]], dim=1)  # [B, 2*enc_hidden]
        c_cnc = torch.cat([enc_c[-2], enc_c[-1]], dim=1)  # [B, 2*enc_hidden]

        dec_h = torch.tanh(self.h_bridge(h_cnc)).unsqueeze(0)  # [1, B, dec_hidden]
        dec_c = torch.tanh(self.c_bridge(c_cnc)).unsqueeze(0)  # [1, B, dec_hidden]

        return enc_output, (dec_h, dec_c)

    def _attend(self, dec_out: torch.Tensor, enc_output: torch.Tensor, src_mask: torch.Tensor):
        """
        dec_out: [B, 1, dec_hidden]
        enc_output: [B, S, enc_out_dim]
        returns context: [B,1,enc_out_dim]
        """
        # Project encoder outputs to dec_hidden for scoring
        proj_enc = self.attn_proj(enc_output)  # [B,S,dec_hidden]

        # scores: [B,1,S]
        scores = torch.bmm(dec_out, proj_enc.transpose(1, 2))

        if src_mask is not None:
            scores = scores.masked_fill(src_mask.unsqueeze(1) == 0, -1e9)

        alphas = F.softmax(scores, dim=-1)  # [B,1,S]
        context = torch.bmm(alphas, enc_output)  # [B,1,enc_out_dim]
        return context, alphas

    def forward(self, src_ids: torch.Tensor, src_mask: torch.Tensor, tgt_in: torch.Tensor):
        """
        src_ids: [B, S]
        src_mask: [B, S]
        tgt_in: [B, T] (teacher forcing input, begins with BOS)
        returns logits: [B, T, V]
        """
        batch_size = src_ids.size(0)
        enc_output, decoder_hidden = self._encode(src_ids)

        tgt_emb = self.dropout(self.tgt_embedding(tgt_in))  # [B, T, E]
        seq_len = tgt_in.size(1)

        outputs = []
        context = torch.zeros(batch_size, 1, self.enc_out_dim, device=src_ids.device)

        for t in range(seq_len):
            input_t = tgt_emb[:, t:t+1, :]  # [B,1,E]
            rnn_input = torch.cat([input_t, context], dim=2)  # [B,1,E+enc_out_dim]

            dec_out, decoder_hidden = self.decoder(rnn_input, decoder_hidden)  # [B,1,dec_hidden]
            context, _ = self._attend(dec_out, enc_output, src_mask)           # [B,1,enc_out_dim]

            concat_input = torch.cat([dec_out, context], dim=2)                # [B,1,dec_hidden+enc_out_dim]
            concat_out = torch.tanh(self.concat(concat_input))                 # [B,1,dec_hidden]
            logits = self.out(concat_out)                                      # [B,1,V]
            outputs.append(logits)

        outputs = torch.cat(outputs, dim=1)  # [B,T,V]
        return outputs

    @staticmethod
    def _get_ngrams(tokens, n: int):
        if len(tokens) < n:
            return set()
        return set(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))

    def generate(
        self,
        src_ids: torch.Tensor,
        src_mask: torch.Tensor,
        max_len: int = 50,
        bos_id: int = None,
        eos_id: int = None,
        no_repeat_ngram_size: int = 3,
        repetition_penalty: float = 1.15,
    ):
        """
        Greedy decoding + anti-repetition:
        - EOS early stop per sample
        - repetition penalty
        - no-repeat ngram blocking

        Returns:
          generated_ids: List[List[int]] (batch of token id sequences INCLUDING BOS and generated tokens)
        """
        if bos_id is None or eos_id is None:
            raise ValueError("bos_id and eos_id must be provided for generation")

        self.eval()
        device = src_ids.device
        B = src_ids.size(0)

        enc_output, decoder_hidden = self._encode(src_ids)

        # start with BOS for each sample
        ys = torch.full((B, 1), bos_id, dtype=torch.long, device=device)  # [B,1]
        context = torch.zeros(B, 1, self.enc_out_dim, device=device)

        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_len):
            # embed last token
            tgt_emb = self.tgt_embedding(ys[:, -1:])  # [B,1,E]
            rnn_input = torch.cat([tgt_emb, context], dim=2)

            dec_out, decoder_hidden = self.decoder(rnn_input, decoder_hidden)  # [B,1,dec_hidden]
            context, _ = self._attend(dec_out, enc_output, src_mask)           # [B,1,enc_out_dim]

            concat_input = torch.cat([dec_out, context], dim=2)
            concat_out = torch.tanh(self.concat(concat_input))
            logits = self.out(concat_out).squeeze(1)  # [B,V]

            # repetition penalty: penalize tokens already generated
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
                        prefix = tuple(prev[-(n-1):])
                        ngrams = self._get_ngrams(prev, n)
                        banned = []
                        # banning all tokens that would recreate an existing n-gram
                        # NOTE: vocab can be large; loop is ok for single-sample interactive,
                        # for batch evaluation it's still acceptable for your sizes.
                        for cand in range(logits.size(-1)):
                            cand_ng = prefix + (cand,)
                            if cand_ng in ngrams:
                                banned.append(cand)
                        if banned:
                            logits[b, banned] = -1e9

            next_token = torch.argmax(logits, dim=-1)  # [B]
            # force EOS for finished sequences
            next_token = torch.where(finished, torch.tensor(eos_id, device=device), next_token)

            ys = torch.cat([ys, next_token.unsqueeze(1)], dim=1)  # append

            finished = finished | (next_token == eos_id)
            if torch.all(finished):
                break

        # Return as python lists per sample
        return [seq.tolist() for seq in ys]
