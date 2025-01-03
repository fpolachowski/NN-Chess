import os
import math
import torch
import torch.nn as nn
from safetensors.torch import load_model, save_model

# Positional Encoding Class
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=300):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class ChessModel(nn.Module):
    def __init__(self,
                num_embeddings,
                transformer_width=512,
                transformer_layers=6,
                transfromer_nheads=8,
                dropout=0.1,
                ) -> None:
        super(ChessModel, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, transformer_width)
        
        encoder_layer = nn.TransformerEncoderLayer(transformer_width, transfromer_nheads, 4*transformer_width, dropout,batch_first=True)
        encoder_norm = nn.LayerNorm(transformer_width)
        self.move_encoder = nn.TransformerEncoder(encoder_layer, transformer_layers, encoder_norm)
        
        self.transformer = nn.Transformer(d_model=transformer_width, 
                       nhead=transfromer_nheads, 
                       num_encoder_layers=transformer_layers, 
                       num_decoder_layers=transformer_layers, 
                       dim_feedforward=4*transformer_width, 
                       dropout=dropout,
                       batch_first=True)
        
        self.positional_encoding = PositionalEncoding(transformer_width, dropout)
        
        # single GPU assumed
        self.use_gpu = False
        self.device = None
        self.gpu_device = torch.device("cuda:0")
        self.cpu_device = torch.device("cpu")
        self.cpu_mode()
        
        self.init_weights()
        
    def init_weights(self) -> None:
        pass
    
    def encode_moves(self, src):
        return self.embedding(src)
        
    def forward(self, src, tgt):
        # Add embeddings and positional encoding
        src = self.embedding(src) * math.sqrt(self.embedding.embedding_dim)
        src = self.positional_encoding(src)
        tgt = self.embedding(tgt) * math.sqrt(self.embedding.embedding_dim)
        tgt = self.positional_encoding(tgt)
        
        # Generate source and target masks
        src_mask = self.transformer.generate_square_subsequent_mask(src.size(1), self.device)
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(1), self.device)

        output = self.transformer(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
        
        return output
    
    def load_checkpoint(self, exp_folder_path, suffix):
        load_model(self, os.path.join(exp_folder_path, "model_{}.safetensors".format(suffix)))
        print("== Checkpoint ({}) is loaded from {}".format(suffix, exp_folder_path))

    def save_checkpoint(self, exp_folder_path, suffix):
        save_model(self, os.path.join(exp_folder_path, "model_{}.safetensors".format(suffix)))
        print("== Checkpoint ({}) is saved to {}".format(suffix, exp_folder_path))

    def cpu_mode(self):
        self.use_gpu = False
        self.to(self.cpu_device)
        self.device = self.cpu_device

    def gpu_mode(self):
        self.use_gpu = True
        self.to(self.gpu_device)
        self.device = self.gpu_device

    def train_mode(self):
        self.train()

    def eval_mode(self):
        self.eval()