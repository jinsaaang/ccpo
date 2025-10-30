import torch
import torch.nn as nn
import torch.fft
import torch.nn.functional as F
# from layers.ts_models_utils import series_decomp, DataEmbedding, Inception_Block_V1

# DLinear
class DLinear(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2205.13504.pdf
    """

    def __init__(self, configs, individual=False):
        """
        individual: Bool, whether shared model among different variates.
        """
        super(DLinear, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        # Series decomposition block from Autoformer
        self.decompsition = series_decomp(configs.moving_avg)
        self.individual = individual
        self.channels = configs.enc_in
        self.dropout = configs.dropout # p=0.1

        # 정규화 레이어 추가 (LayerNorm)
        self.norm = nn.LayerNorm(self.channels)

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()

            for i in range(self.channels):
                self.Linear_Seasonal.append(
                    nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend.append(
                    nn.Linear(self.seq_len, self.pred_len))

                self.Linear_Seasonal[i].weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
                self.Linear_Trend[i].weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

            self.Linear_Seasonal.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
            self.Linear_Trend.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))

        # self._init_weights()

    def _init_weights(self):
        # Linear 레이어 가중치 초기화 (Xavier)
        if self.individual:
            for i in range(self.channels):
                nn.init.xavier_uniform_(self.Linear_Seasonal[i].weight)
                nn.init.zeros_(self.Linear_Seasonal[i].bias)
                nn.init.xavier_uniform_(self.Linear_Trend[i].weight)
                nn.init.zeros_(self.Linear_Trend[i].bias)
        else:
            nn.init.xavier_uniform_(self.Linear_Seasonal.weight)
            nn.init.zeros_(self.Linear_Seasonal.bias)
            nn.init.xavier_uniform_(self.Linear_Trend.weight)
            nn.init.zeros_(self.Linear_Trend.bias)

    def encoder(self, x):
        x = self.norm(x)
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(
            0, 2, 1), trend_init.permute(0, 2, 1)
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
                                          dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), self.pred_len],
                                       dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](
                    seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](
                    trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            seasonal_output = F.dropout(seasonal_output, p=self.dropout, training=self.training)
            trend_output = self.Linear_Trend(trend_init)
            trend_output = F.dropout(trend_output, p=self.dropout, training=self.training)

        x = seasonal_output + trend_output
        return x.permute(0, 2, 1)

    def forecast(self, x_enc):
        # Encoder
        return self.encoder(x_enc)

    def forward(self, x_enc, mask=None):
        dec_out = self.forecast(x_enc)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]
    

# LSTM
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


# MLP
class MLP(nn.Module):
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
    