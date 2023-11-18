import math
import torch
import torch.nn as  nn
import torch.nn.functional as F
import transformers

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MultiHeadedAttention(nn.Module):
    def __init__(self, num_heads, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % num_heads == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        
        self.dense = nn.Linear(d_model, d_model)
        
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def attention(self, query, key, value):
        "Compute 'Scaled Dot Product Attention'"
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(d_k)
        # attention weights
        p_attn = F.softmax(scores, dim = -1)
        return torch.matmul(p_attn, value), p_attn
        
    def forward(self, query, key, value):
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query = self.wq(query).view(nbatches, -1, self.num_heads, self.d_k)
        query = query.transpose(1, 2)
        
        key = self.wk(key).view(nbatches, -1, self.num_heads, self.d_k)
        key = key.transpose(1, 2)
        
        value = self.wv(value).view(nbatches, -1, self.num_heads, self.d_k)
        value = value.transpose(1, 2)
        
        #        query, key, value = \
        #            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        #             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = self.attention(query, key, value)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.num_heads * self.d_k)
             
        x = self.dense(x)
        x = self.dropout(x)
        
        return x

class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, d_model, num_heads, dropout):
        super(DecoderLayer, self).__init__()
        self.mha = MultiHeadedAttention(num_heads, d_model, dropout)
        self.layernorm = nn.LayerNorm(d_model, eps=1e-6)
 
    def forward(self, s, k):
        attn_output = self.mha(k, s, s) # q,k,v
        s = self.layernorm(s + attn_output)
        return s


class Decoder(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, dropout):
        super(Decoder, self).__init__()
        
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, dropout) 
                                            for _ in range(num_layers)])
        
    def forward(self, s, k):
        for dec_layer in self.layers:
            s = dec_layer(s, k)
        return s



class SC_SKB(nn.Module):
    def __init__(self, num_vocab):
        super(SC_SKB, self).__init__()
        self.embedding_s = nn.Embedding(num_vocab, 128)
        self.embedding_k1 = nn.Embedding(num_vocab, 128)
        self.embedding_k2 = nn.Embedding(num_vocab, 128)

        encode_layer = nn.TransformerEncoderLayer(d_model=128, nhead=8, batch_first=True, dropout=0.1, activation='gelu')

        self.semantic_extraction_s = nn.TransformerEncoder(encoder_layer=encode_layer, num_layers=1)
        self.semantic_extraction_k1 = nn.TransformerEncoder(encoder_layer=encode_layer, num_layers=1)
        self.semantic_extraction_k2 = nn.TransformerEncoder(encoder_layer=encode_layer, num_layers=1)
        
        self.decode_model_t = Decoder(d_model=128, num_heads=8, num_layers=1, dropout=0.1)

        self.channel_encode = nn.Sequential(nn.Linear(128, 32), nn.PReLU(), nn.Linear(32, 12))

        self.channel_decode = nn.Sequential(nn.Linear(12, 32), nn.PReLU(), nn.Linear(32, 128))

        self.decode_model_r = Decoder(d_model=128, num_heads=8, num_layers=1, dropout=0.1)

        self.semantic_recover = nn.TransformerEncoder(encoder_layer=encode_layer, num_layers=1)

        self.dense = nn.Linear(128, num_vocab)


class Channels():
    def PowerNormalize(self, x):
        x_square = torch.mul(x, x)
        power = math.sqrt(2) * x_square.mean(dim=1, keepdim=True).sqrt()
        out = torch.div(x, power)
        return out

    def SNR_to_noise(self, snr):
        snr = 10 ** (snr / 10)
        noise_std = 1 / ((2 * snr)**0.5)
        return noise_std       
        
    def AWGN(self, Tx_sig, snr):
        n_var = self.SNR_to_noise(snr)
        Rx_sig = self.PowerNormalize(Tx_sig)
        Rx_sig = Rx_sig + torch.normal(0, n_var, size=Rx_sig.shape, device=device)
        return Rx_sig

    def Rayleigh(self, Tx_sig, snr):
        shape = Tx_sig.shape # (B, 2M)
        H_real = torch.normal(0, math.sqrt(1/2), size=[1]).to(device)
        H_imag = torch.normal(0, math.sqrt(1/2), size=[1]).to(device)
        H = torch.Tensor([[H_real, -H_imag], [H_imag, H_real]]).to(device)
        n_var = self.SNR_to_noise(snr)
        Tx_sig = self.PowerNormalize(Tx_sig)

        Tx_sig = torch.matmul(Tx_sig.view(shape[0], -1, 2), H) # (B, M, 2) (2, 2)
        Rx_sig = Tx_sig + torch.normal(0, n_var, size=Tx_sig.shape, device=device)
        # Channel estimation
        Rx_sig = torch.matmul(Rx_sig, torch.inverse(H)).view(shape)
        return Rx_sig

    def Rician(self, Tx_sig, snr, K_dB=0):
        shape = Tx_sig.shape
        K = 10**(K_dB/10)
        mean = math.sqrt(K / (K + 1))
        std = math.sqrt(1 / (K + 1)) * math.sqrt(1/2)
        H_real = torch.normal(mean, std, size=[1]).to(device)
        H_imag = torch.normal(0, std, size=[1]).to(device)
        H = torch.Tensor([[H_real, -H_imag], [H_imag, H_real]]).to(device)
        n_var = self.SNR_to_noise(snr)
        Tx_sig = self.PowerNormalize(Tx_sig)

        Tx_sig = torch.matmul(Tx_sig.view(shape[0], -1, 2), H)
        Rx_sig = Tx_sig + torch.normal(0, n_var, size=Tx_sig.shape, device=device)
        # Channel estimation
        Rx_sig = torch.matmul(Rx_sig, torch.inverse(H)).view(shape)
        return Rx_sig

