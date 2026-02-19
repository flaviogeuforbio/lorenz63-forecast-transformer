import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import math

MAX_LEN = 2048 #maximum context lenght

#classic sinusoidal positional encoding (vaswani et al.)
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = MAX_LEN):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  #(T, d)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  #(T,1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )  #(d/2,)

        pe[:, 0::2] = torch.sin(position * div_term)  #even
        pe[:, 1::2] = torch.cos(position * div_term)  #odd
        pe = pe.unsqueeze(0)  #(1, T, d)

        #Register as buffer (not a parameter, but moves with device)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, d_model)
        """
        T = x.size(1)
        return x + self.pe[:, :T, :]
    

class MultiHeadSelfAttention(nn.Module):
    #Multi-head self attention:
    # INPUT: [B, T, d_model] -> ATTENTION MAPS: [B, n_heads, T, T] -> OUTPUT: [B, T, d_model]

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()

        #saving hyperparameters
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads #dimension of each head

        if d_model % n_heads != 0: #check d_model = n_heads * d_heads
            raise ValueError(f"{d_model} must be divisible by {n_heads}!")
        
        #initializing model parameters
        self.qkv = nn.Linear(d_model, 3 * d_model, bias = True) #contains q, k, v projections together (more efficient)
        self.out = nn.Linear(d_model, d_model, bias = True) #final projection after self-attention
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        #extracting input dimensions
        B, T, _ = x.shape

        #calculating q, k, v 
        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, chunks = 3, dim = -1) #[B, T, d_model]

        #chunking q, k, v to get queries, values and keys for each head
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2) #[B, n_heads, T, d_head]
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2) # same dim.
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2) # same dim.

        #getting attention maps
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head) #[B, n_heads, T, T]
        attn = torch.softmax(attn, dim=-1) #each row sums up to 1
        attn = self.dropout(attn) #applying dropout to attention maps (helps for generalization)

        #output values + concatenation btw heads
        y = torch.matmul(attn, v) #[B, n_heads, T, d_head]
        y = y.transpose(1, 2).contiguous().view(B, T, self.d_model) #[B, T, d_model]
        y = self.out(y) #final output projection

        return (y, attn) if return_attn else (y, None)
    

class LorenzTransformerEncoder(nn.Module):
    #Transfomer Encoder Block:
    #INPUT: [B, T, d_model] -> LN -> MULTI-HEAD SA -> LN -> FFN: [B, T, d_model]
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        #initializing MHSA + LN layers
        self.attn = MultiHeadSelfAttention(d_model = d_model, n_heads = n_heads, dropout = dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        #initializing feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        #MHSA
        h = self.norm1(x)
        a, attn = self.attn(h, return_attn=return_attn)
        x = x + a

        #FFN
        h = self.norm2(x)
        x = x + self.ff(h)

        return x, attn #x: [B, T, d_model], attn: [B, n_heads, T, T]
    

#defining hyperparameters of the Transfomer
@dataclass
class LorenzTransformerConfig:
    seq_len: int = 128 #n. of tokens per batch
    n_layers: int = 4 #n. of encoder layers
    d_model: int = 128 #embedding dimension
    H: int = 32 #forecasting window dimension
    n_heads: int = 4 #n. of heads in MHSA
    d_ff: int = 512 #proj. dimension in ffn
    dropout: float = 0.1 #dropout hyp.
    use_posenc: bool = True #if True, encoding = in_proj + pos_enc


class LorenzTransformer(nn.Module):
    #Whole transformer architecture: ENCODER + TRANSFORMER BLOCKS + 2 HEADS (1 for regime classification, 1 for rho regression)

    def __init__(self, cfg: LorenzTransformerConfig):
        super().__init__()
        self.cfg = cfg #saving configuration object

        #encoding
        self.pe = SinusoidalPositionalEncoding(d_model = cfg.d_model, max_len = cfg.seq_len) if cfg.use_posenc else nn.Identity()
        self.in_proj = nn.Linear(3, cfg.d_model)
        
        self.dropout = nn.Dropout(cfg.dropout)

        #transfomer blocks
        self.blocks = nn.ModuleList([
            LorenzTransformerEncoder(
                d_model = cfg.d_model,
                n_heads = cfg.n_heads,
                d_ff = cfg.d_ff,
                dropout = cfg.dropout
            ) for _ in range(cfg.n_layers)
        ]
        )

        # #pho regression head
        # self.rho_head = nn.Sequential(
        #     nn.LayerNorm(cfg.d_model),
        #     nn.Linear(cfg.d_model, 1)
        # )

        # #regime classification head
        # self.regime_head = nn.Sequential(
        #     nn.LayerNorm(cfg.d_model),
        #     nn.Linear(cfg.d_model, 1)
        # )

        #forecast head
        self.forecast_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model), 
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.H*3)
        )


    def forward(self, x: torch.Tensor, return_attn: bool = False):
        #check input dimensionality
        B, T, C = x.shape
        if T > MAX_LEN:
            raise ValueError(f"Sequence lenght ({T}) > max value ({self.cfg.seq_len})")
        
        if C != 3:
            raise ValueError(f"Expected input features = 3 (x,y,z), got {C}")

        #encoding input tensor
        h = self.pe(self.in_proj(x))
        h = self.dropout(h)

        #encoder blocks
        attn_maps = [] if return_attn else None #collecting attention maps if requested
        for block in self.blocks:
            h, attn = block(h, return_attn=return_attn)
            if return_attn:
                attn_maps.append(attn)


        # #pooling of output data (averaging over sequence dimension)
        # pooled = torch.mean(h, dim = 1) #h: [B, T, d_model] -> pooled: [B, d_model]

        #picking last temporal token
        last_token = h[:, -1, :] # [B, T, d_model] -> [B, d_model]

        #final heads
        # rho_pred = self.rho_head(pooled)
        # regime_logit = self.regime_head(pooled)
        forecast_traj = self.forecast_head(last_token) # [B, d_model] -> [B, H*3]
        forecast_traj = forecast_traj.reshape(B, self.cfg.H, 3) #[B, H*3] -> [B, H, 3] -- output forecast sequence

        return forecast_traj, attn_maps
        











