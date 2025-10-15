import torch as th
from torch import nn

class MLP(th.nn.Module):
    """
    시계열 예측용 MLP 베이스라인 (OR 저널 스타일 + 적절한 용량).
    입력: [B, seq_len, feature_dim]
    출력: [B, pred_len, feature_dim]
    """
    def __init__(self, args):
        super(MLP, self).__init__()
        seq_len = args.seq_len
        pred_len = args.pred_len
        feature_dim = args.enc_in
        hidden_dim = 128
        dropout = getattr(args, "dropout", 0.1)

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.feature_dim = feature_dim

        # Input normalization
        self.input_norm = nn.LayerNorm(seq_len * feature_dim)
        
        self.flatten = nn.Flatten()
        self.mlp = nn.Sequential(
            # First layer
            nn.Linear(seq_len * feature_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),  # BatchNorm → LayerNorm (더 안정적)
            nn.GELU(),  # ReLU → GELU (더 부드러운 활성화)
            nn.Dropout(dropout),
            
            # Second layer  
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            
            # Third layer (추가)
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            
            # Output layer
            nn.Linear(hidden_dim // 2, pred_len * feature_dim)
        )

        self._init_weights()

    def _init_weights(self):
        """Xavier 초기화 (GELU와 잘 맞음)"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, *args, **kwargs):
        # x: [B, seq_len, feature_dim]
        batch_size = x.size(0)
        
        # Flatten and normalize
        x = self.flatten(x)  # [B, seq_len * feature_dim]
        x = self.input_norm(x)  # Input normalization
        
        # MLP forward
        out = self.mlp(x)    # [B, pred_len * feature_dim]
        
        # Reshape to output format
        out = out.view(batch_size, self.pred_len, self.feature_dim)  # [B, pred_len, feature_dim]
        
        return out