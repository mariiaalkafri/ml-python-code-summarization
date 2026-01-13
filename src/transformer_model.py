import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1) # [max_len, 1, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [S, B, E]
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerSeq2Seq(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_encoder_layers=4, num_decoder_layers=4, dim_feedforward=1024, dropout=0.1, pad_id=0):
        super(TransformerSeq2Seq, self).__init__()
        
        self.d_model = d_model
        self.pad_id = pad_id
        
        self.src_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.tgt_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.pos_decoder = PositionalEncoding(d_model, dropout)
        
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, src_ids, tgt_ids, src_key_padding_mask=None, tgt_key_padding_mask=None, tgt_mask=None):
        # src_ids: [B, S]
        # tgt_ids: [B, T]
        # nn.Transformer expects [S, B, E] by default unless batch_first=True is set. 
        # But we didn't set batch_first=True in nn.Transformer (default is False), so we transpose inputs.
        
        src_emb = self.src_embedding(src_ids) * math.sqrt(self.d_model) # [B, S, E]
        tgt_emb = self.tgt_embedding(tgt_ids) * math.sqrt(self.d_model) # [B, T, E]
        
        src_emb = src_emb.transpose(0, 1) # [S, B, E]
        tgt_emb = tgt_emb.transpose(0, 1) # [T, B, E]
        
        src_emb = self.pos_encoder(src_emb)
        tgt_emb = self.pos_decoder(tgt_emb)
        
        output = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
            # memory_key_padding_mask=src_key_padding_mask # usually same as src_key_padding_mask
        )
        
        # output: [T, B, E] -> transpose back to [B, T, E] required?
        # The loss function usually expects [B, C, T] or [B, T, C]. Let's return logits as [B, T, V]
        
        output = output.transpose(0, 1) # [B, T, E]
        logits = self.out(output) # [B, T, V]
        
        return logits

    def generate(self, src_ids, src_mask, max_len=50, bos_id=None, eos_id=None):
        # src_ids: [B, S]
        # src_mask: [B, S] (1=valid, 0=pad) -> convert to bool (True=pad) if needed
        
        device = src_ids.device
        batch_size = src_ids.size(0)
        
        # Create src_key_padding_mask for encoder
        # src_mask is 1 for valid, 0 for pad. 
        # nn.Transformer expects True for pad.
        src_key_padding_mask = (src_mask == 0)
        
        src_emb = self.src_embedding(src_ids) * math.sqrt(self.d_model)
        src_emb = src_emb.transpose(0, 1) # [S, B, E]
        src_emb = self.pos_encoder(src_emb)
        
        memory = self.transformer.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)
        
        # Init decode with <bos>
        decoder_input = torch.tensor([[bos_id]], device=device).repeat(batch_size, 1) # [B, 1]
        
        generated_ids = []
        finished_sequences = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        for i in range(max_len):
            tgt_emb = self.tgt_embedding(decoder_input) * math.sqrt(self.d_model)
            tgt_emb = tgt_emb.transpose(0, 1) # [T_curr, B, E]
            tgt_emb = self.pos_decoder(tgt_emb)
            
            tgt_mask = self.transformer.generate_square_subsequent_mask(tgt_emb.size(0)).to(device)
            
            output = self.transformer.decoder(
                tgt_emb, 
                memory, 
                tgt_mask=tgt_mask,
                memory_key_padding_mask=src_key_padding_mask
            )
            
            output = output.transpose(0, 1) # [B, T_curr, E]
            logits = self.out(output[:, -1, :]) # [B, V] (last step only)
            
            next_token = logits.argmax(dim=-1) # [B]
            
            # Simple greedy: append next token
            # In a real batched inference, we need to handle finished sequences, but for now simple loop
            
            # For purely greedy generation one by one
            # Note: This is slightly inefficient as we re-process the whole sequence.
            # Transformer decoder usually caches, but nn.Transformer pure functional doesn't expose cache easily.
            # For this assignment, re-running is acceptable or we'd need customized decoder loop.
            
            # Just taking the first batch item for result if we assume batch=1 in generation commonly in this project
            # But let's support batch structure
            
            col_ids = next_token.unsqueeze(1) # [B, 1]
            decoder_input = torch.cat([decoder_input, col_ids], dim=1)
            
            # We only really care about the newly generated token
            generated_ids.append(next_token) 
            
            # Check eos
            is_eos = (next_token == eos_id)
            finished_sequences = finished_sequences | is_eos
            
            if finished_sequences.all():
                break
                
        # Stack produced tokens: list of [B] tensors -> [B, T_gen]
        generated_ids = torch.stack(generated_ids, dim=1)
        
        # If batch_size=1, return list[int]
        if batch_size == 1:
            return generated_ids[0].tolist()
            
        return generated_ids.tolist()
