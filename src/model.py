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
        
        proj_enc = self.project(encoder_outputs) # [B, S, H]
        attn_scores = torch.bmm(hidden, proj_enc.transpose(1, 2)) # [B, 1, S]
        
        if src_mask is not None:
            mask = src_mask.unsqueeze(1)
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
            
        attn_weights = F.softmax(attn_scores, dim=-1) # [B, 1, S]
        context = torch.bmm(attn_weights, encoder_outputs) # [B, 1, H]
        
        return context, attn_weights

class Seq2SeqLSTMAttn(nn.Module):
    def __init__(self, vocab_size, emb_dim=256, enc_hidden=256, dec_hidden=512, num_layers=1, dropout=0.2, pad_id=0):
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
        
        context = torch.zeros(batch_size, 1, self.enc_out_dim).to(src_ids.device)
        
        for t in range(seq_len):
            input_t = tgt_emb[:, t:t+1, :] 
            rnn_input = torch.cat([input_t, context], dim=2) 
            
            dec_out, decoder_hidden = self.decoder(rnn_input, decoder_hidden) 
            context, attn_weights = self.attention(dec_out, enc_output, src_mask) 
            
            concat_input = torch.cat([dec_out, context], dim=2) 
            concat_out = torch.tanh(self.concat(concat_input)) 
            
            logits = self.out(concat_out) 
            outputs.append(logits)
            
        outputs = torch.cat(outputs, dim=1) 
        return outputs

    def generate(self, src_ids, src_mask, max_len=50, bos_id=None, eos_id=None):
        # src_ids: [1, S] (single batch for simplicity or batch)
        # src_mask: [1, S]
        
        batch_size = src_ids.size(0)
        
        # Encode
        src_emb = self.dropout(self.src_embedding(src_ids)) 
        enc_output, (enc_h, enc_c) = self.encoder(src_emb) 
        
        # Init Decoder State
        h_cnc = torch.cat([enc_h[-2], enc_h[-1]], dim=1) 
        c_cnc = torch.cat([enc_c[-2], enc_c[-1]], dim=1) 
        
        dec_h = torch.tanh(self.h_bridge(h_cnc)).unsqueeze(0) 
        dec_c = torch.tanh(self.c_bridge(c_cnc)).unsqueeze(0) 
        
        decoder_hidden = (dec_h, dec_c)
        
        # Init Input (<bos>)
        if bos_id is None:
            raise ValueError("bos_id must be provided for generation")
            
        decoder_input = torch.tensor([[bos_id]], device=src_ids.device).repeat(batch_size, 1) # [B, 1]
        
        context = torch.zeros(batch_size, 1, self.enc_out_dim).to(src_ids.device)
        
        generated_ids = []
        
        for t in range(max_len):
            tgt_emb = self.dropout(self.tgt_embedding(decoder_input)) # [B, 1, E]
            
            rnn_input = torch.cat([tgt_emb, context], dim=2) 
            dec_out, decoder_hidden = self.decoder(rnn_input, decoder_hidden) 
            
            context, attn_weights = self.attention(dec_out, enc_output, src_mask) 
            
            concat_input = torch.cat([dec_out, context], dim=2) 
            concat_out = torch.tanh(self.concat(concat_input)) 
            
            logits = self.out(concat_out) # [B, 1, V]
            
            # Greedy search
            next_token = logits.argmax(dim=-1) # [B, 1]
            generated_ids.append(next_token.item())
            
            decoder_input = next_token
            
            if next_token.item() == eos_id:
                break
                
        return generated_ids
