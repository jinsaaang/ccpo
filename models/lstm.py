import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.input_size = configs.enc_in
        self.hidden_size = 96  # 조금 줄임
        self.num_layers = 2  # 레이어 수도 줄임
        self.output_size = configs.c_out
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.dropout = min(configs.dropout * 1.5, 0.3)  # 드롭아웃
        
        # 입력 정규화
        self.input_norm = nn.LayerNorm(self.input_size)
        
        # 단방향 LSTM
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0,
            bidirectional=False
        )
        
        # LSTM 출력 정규화
        self.lstm_norm = nn.LayerNorm(self.hidden_size)
        
        # 더 강한 정규화를 가진 projection head
        self.projection = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.LayerNorm(self.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(self.dropout)
        )
        
        # Final output layer
        self.output_layer = nn.Linear(self.hidden_size // 2, self.output_size)
        
        # Weight initialization
        self._init_weights()

    def _init_weights(self):
        # LSTM weight initialization
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # Forget gate bias를 1로 설정
                n = param.size(0)
                param.data[(n//4):(n//2)].fill_(1)
        
        # Linear layer initialization
        for module in [self.projection, self.output_layer]:
            if isinstance(module, nn.Sequential):
                for layer in module:
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_uniform_(layer.weight)
                        nn.init.zeros_(layer.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None, y=None):
        # x_enc: [B, seq_len, input_size]
        batch_size = x_enc.size(0)
        
        # Input normalization
        x = self.input_norm(x_enc)
        
        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # LSTM output normalization
        lstm_out = self.lstm_norm(lstm_out)
        
        # 예측 구간만 추출
        if self.pred_len == 1:
            out = lstm_out[:, -1:, :]  # [B, 1, hidden_size]
        else:
            out = lstm_out[:, -self.pred_len:, :]  # [B, pred_len, hidden_size]
        
        # Projection (residual connection 제거)
        out = self.projection(out)
        
        # Final output
        out = self.output_layer(out)
        
        return out  # [B, pred_len, output_size]

    
