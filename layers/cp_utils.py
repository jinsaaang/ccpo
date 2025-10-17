import numpy as np
import math
import time
import warnings
import torch
import torch.nn as nn
from torch.utils.data import Subset, TensorDataset, DataLoader
import os
from models.mlp import MLP
from models.dlinear import DLinear
from models.lstm import LSTMModel
import torch.optim as optim

def ellipsoid_volume(covariance_matrix, r):
    # Only compute volume of the ellipsoid along the first r dimensions
    # where r is the number of dimensions with non-zero eigenvalues
    eigenvalues = np.linalg.eigvals(covariance_matrix)
    eigenvalues = np.real(np.sort(eigenvalues)[::-1])
    eps = 1e-6
    num_r = np.sum(eigenvalues > eps)
    det_sigma = np.prod(eigenvalues[:num_r])
    constant_cd = np.pi**(num_r/2) / math.gamma(num_r/2 + 1)  # Volume constant for d-dimensional sphere
    volume = constant_cd * r**num_r * np.sqrt(det_sigma)
    return volume


#### From utils_EnbPI ####
def adjust_alpha_t(alpha_t, alpha, errs, gamma=0.005, method='simple'):
    if method == 'simple':
        # Eq. (2) of Adaptive CI
        return alpha_t+gamma*(alpha-errs[-1])
    else:
        # Eq. (3) of Adaptive CI with particular w_s as given
        t = len(errs)
        errs = np.array(errs)
        w_s_ls = np.array([0.95**(t-i) for i in range(t)]
                          )  # Furtherest to Most recent
        return alpha_t+gamma*(alpha-w_s_ls.dot(errs))


def ave_cov_width(df, Y):
    coverage_res = ((np.array(df['lower']) <= Y) & (
        np.array(df['upper']) >= Y)).mean()
    print(f'Average Coverage is {coverage_res}')
    width_res = (df['upper'] - df['lower']).mean()
    print(f'Average Width is {width_res}')
    return [coverage_res, width_res]

#### Miscellaneous ####


window_size = 300


def rolling_avg(x, window=window_size):
    return np.convolve(x, np.ones(window)/window)[(window-1):-window]


def generate_bootstrap_samples(n, m, B):
    '''
      Return: B-by-m matrix, where row b gives the indices for b-th bootstrap sample
    '''
    samples_idx = np.zeros((B, m), dtype=int)
    for b in range(B):
        sample_idx = np.random.choice(n, m)
        samples_idx[b, :] = sample_idx
    return(samples_idx)


def make_bootstrap_loader(dataset, B=30, replace=True, batch_size=64):
    """
    B: num bootstrap models
    return: List of DataLoader for each bootstrap subset
    """
    
    T = len(dataset)  
    bootstrap_loaders = []

    for b in range(B):
        sampled_indices = np.random.choice(T, size=T, replace=replace)
        subset = Subset(dataset, sampled_indices)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
        bootstrap_loaders.append((loader, set(sampled_indices)))

    return bootstrap_loaders


def strided_app(a, L, S):
    nrows = ((a.shape[0] - L) // S) + 1
    shape = (nrows, L) + a.shape[1:]
    strides = (S * a.strides[0],) + a.strides
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)



def binning(past_resid, cov_mat_est, alpha, bins = 5):
    '''
    Input:
        past residuals: evident
        alpha: signifance level
    Output:
        beta_hat_bin as argmin of the difference
    Description:
        Compute the beta^hat_bin from past_resid, by breaking [0,alpha] into bins (like 20). It is enough for small alpha
        number of bins are determined rather automatic, relative the size of whole domain
    '''
    beta_ls = np.linspace(start=0, stop=alpha, num=bins)
    sizes = np.zeros(bins)
    for i in range(bins):
        width = np.percentile(past_resid, math.ceil(100 * (1 - alpha + beta_ls[i]))) - \
            np.percentile(past_resid, math.ceil(100 * beta_ls[i]))
        sizes[i] = ellipsoid_volume(cov_mat_est, width)
    i_star = np.argmin(sizes)
    return beta_ls[i_star]


def binning_use_RF_quantile_regr(quantile_regr, cov_mat_est, Xtrain, Ytrain, feature, beta_ls, sample_weight=None):
    # API ref: https://sklearn-quantile.readthedocs.io/en/latest/generated/sklearn_quantile.RandomForestQuantileRegressor.html
    feature = feature.reshape(1, -1)
    low_high_pred = quantile_regr.fit(Xtrain, Ytrain, sample_weight).predict(feature)
    num_mid = int(len(low_high_pred)/2)
    low_pred, high_pred = low_high_pred[:num_mid], low_high_pred[num_mid:]
    width = (high_pred-low_pred).flatten()
    width = [ellipsoid_volume(cov_mat_est, w) for w in width]
    i_star = np.argmin(width)
    wid_left, wid_right = low_pred[i_star], high_pred[i_star]
    return i_star, beta_ls[i_star], wid_left, wid_right


def train_models(model_cls, data_loader, valid_data_loader, EPOCHS=100, lr=1e-3, path='./weights/', patience=10, valid_mode=False):
    os.makedirs(path, exist_ok=True)

    models = []
    indices_ls = []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    criterion = nn.MSELoss()
    
    for i, (loader_b, indices_b) in enumerate(data_loader):
        
        sample_x, sample_y, _, _ = next(iter(loader_b))

        if model_cls == MLP:
            input_dim = sample_x.shape[1] * sample_x.shape[2]
            hidden_dim = 2000
            output_dim = sample_y.shape[1] * sample_y.shape[2]
            model_b = model_cls(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim).to(device)
        
        elif model_cls == DLinear:
            seq_len = sample_x.shape[1]
            pred_len = sample_y.shape[1]
            enc_in = sample_x.shape[2]
            
            class DLinearConfig:
                def __init__(self, seq_len, pred_len, enc_in, individual=False):
                    self.seq_len = seq_len
                    self.pred_len = pred_len
                    self.enc_in = enc_in
                    self.individual = individual
                    
            dlinear_configs = DLinearConfig(seq_len=seq_len, pred_len=pred_len, enc_in=enc_in)      # 수정
            model_b = model_cls(dlinear_configs).to(device)
        
        elif model_cls == LSTMModel:
            seq_len = sample_x.shape[1]
            pred_len = sample_y.shape[1]
            enc_in = sample_x.shape[2]
            c_out = sample_y.shape[2]
            
            class LSTMConfig:
                def __init__(self, enc_in, c_out, seq_len, pred_len, dropout):
                    self.enc_in = enc_in
                    self.c_out = c_out
                    self.seq_len = seq_len
                    self.pred_len = pred_len
                    self.dropout = dropout
                    
            lstm_configs = LSTMConfig(enc_in=enc_in, c_out=c_out, seq_len=seq_len, pred_len=pred_len, dropout=0.1)
            model_b = model_cls(lstm_configs).to(device)
            
        else:
            raise ValueError(f"Unsupported model class: {model_cls}")
            
        optimizer = optim.Adam(model_b.parameters(), lr=lr)
        
        best_val_loss = float('inf')
        patience_counter = 0
        

        model_name = model_cls.__name__ 
        model_save_path = f"{path}/{model_name}_model_b{i}.pt"

        for epoch in range(EPOCHS):
            model_b.train()
            total_train_loss = 0.0 
            for X_batch, y_batch, _, _ in loader_b:
                
                if model_cls == MLP:
                    X_batch = X_batch.float().to(device).view(X_batch.size(0), -1)
                    y_batch = y_batch.float().to(device).view(y_batch.size(0), -1)
                else:
                    X_batch = X_batch.float().to(device)
                    y_batch = y_batch.float().to(device)
                
                optimizer.zero_grad()
                preds = model_b(X_batch)
                loss = criterion(preds, y_batch)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()
            
            avg_train_loss = total_train_loss / len(loader_b)

            if valid_mode:
                model_b.eval()
                total_val_loss = 0.0
                with torch.no_grad():
                    for X_val, y_val, _, _ in valid_data_loader:
                        if model_cls == MLP:
                            X_val = X_val.float().to(device).view(X_val.size(0), -1)
                            y_val = y_val.float().to(device).view(y_val.size(0), -1)
                        else:
                            X_val = X_val.float().to(device)
                            y_val = y_val.float().to(device)
                        
                        preds_val = model_b(X_val)
                        val_loss = criterion(preds_val, y_val)
                        total_val_loss += val_loss.item()
                
                avg_val_loss = total_val_loss / len(valid_data_loader)
                
                print(f"Model Num {i} ({model_name}), Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Valid Loss: {avg_val_loss:.4f}")

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    torch.save(model_b.state_dict(), model_save_path)
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    print(f"Early stopping for model {i} ({model_name}) at epoch {epoch+1}!")
                    break
            else:
                print(f"Model Num {i} ({model_name}), Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}")
        
        if valid_mode:
            if os.path.exists(model_save_path):
                print(f"Loading best model for {model_name}_{i} with validation loss: {best_val_loss:.4f}")
                model_b.load_state_dict(torch.load(model_save_path))
            else:
                print(f"Warning: No model was saved for {model_name}_{i} as validation loss never improved.")
        else:
            print(f"Saving final model for {model_name}_{i} after {EPOCHS} epochs.")
            torch.save(model_b.state_dict(), model_save_path)
        
        models.append(model_b)
        indices_ls.append(indices_b)
        print("-" * 60)
        
    return models, indices_ls


# def compute_residuals(model_type, valid_loader, test_loader, models, device="cpu"):
#     """
#     models: 부트스트랩 학습된 모델 리스트
#     model_type: ["MLP" | "DLinear" | "LSTM"]
#     반환: {
#       'valid': {'y_true', 'y_pred', 'resid'},
#       'test' : {'y_true', 'y_pred', 'resid'},
#       'raw'  : {'y_valid_scaled','yhat_valid_scaled','y_test_scaled','yhat_test_scaled'}
#     }
#     """
#     device = torch.device(device if torch.cuda.is_available() else "cpu")

#     def prep_inputs(X, y):
#         if model_type == "MLP":
#             return X.float().to(device).view(X.size(0), -1), y.float().to(device)
#         return X.float().to(device), y.float().to(device)

#     def inverse(T, dataset):
#         if hasattr(dataset, "inverse_transform") and callable(dataset.inverse_transform):
#             Tnp = T.detach().cpu().numpy()  # [N, L, d]
#             N, L, D = Tnp.shape
#             try:
#                 inv_np = dataset.inverse_transform(Tnp)
#                 inv = torch.tensor(inv_np, dtype=T.dtype)
#                 if inv.shape == T.shape:  # 3D 지원
#                     return inv
#             except Exception:
#                 pass
#             inv_slices = []
#             for t in range(L):
#                 inv_slices.append(dataset.inverse_transform(Tnp[:, t, :]))  # [N, d]
#             inv_np = np.stack(inv_slices, axis=1)  # [N, L, d]
#             return torch.tensor(inv_np, dtype=T.dtype)
#         return T  # 역변환 불가 시 그대로


#     def gather_targets(loader):
#         ys = []
#         for X, y, _, _ in loader:
#             ys.append(y)
#         return torch.cat(ys, dim=0)  # [N, L, d]

#     valid_ds = valid_loader.dataset
#     test_ds  = test_loader.dataset
    
#     Yv = gather_targets(valid_loader)   # [N_valid, L, d] (CPU)
#     Yt = gather_targets(test_loader)    # [N_test,  L, d]

#     Pv_list, Pt_list = [], []
#     with torch.no_grad():
#         for m in models:
#             m.eval(); m.to(device)

#             outs_v, outs_t = [], []
            
#             for Xb, yb, _, _ in valid_loader:
#                 Xb, _ = prep_inputs(Xb, yb)
#                 outs_v.append(m(Xb).detach().cpu())
                
#             for Xb, yb, _, _ in test_loader:
#                 Xb, _ = prep_inputs(Xb, yb)
#                 outs_t.append(m(Xb).detach().cpu())

#             Pv_list.append(torch.cat(outs_v, dim=0))  # [N_valid, L, d]
#             Pt_list.append(torch.cat(outs_t, dim=0))  # [N_test,  L, d]

#     Pv = torch.stack(Pv_list, dim=0).mean(dim=0)  # [N_valid, L, d]
#     Pt = torch.stack(Pt_list, dim=0).mean(dim=0)  # [N_test,  L, d]

#     # 3) inverse_transform (가능하면)
#     Yv_inv = inverse(Yv, valid_loader)
#     Pv_inv = inverse(Pv, valid_loader)
#     Yt_inv = inverse(Yt, test_loader)
#     Pt_inv = inverse(Pt, test_loader)

#     # Compute residuals
#     Rv = (Yv_inv - Pv_inv)
#     Rt = (Yt_inv  - Pt_inv)

#     return {
#         "valid": {"y_true": Yv_inv, "y_pred": Pv_inv, "resid": Rv},
#         "test":  {"y_true": Yt_inv, "y_pred": Pt_inv, "resid": Rt},
#         "raw":   {
#             "y_valid_scaled": Yv, "yhat_valid_scaled": Pv,
#             "y_test_scaled":  Yt, "yhat_test_scaled":  Pt,
#         },
#     }

def compute_residuals(model_type, valid_loader, test_loader, models, loader, device="cpu"): # [수정] loader 객체 추가
    """
    models: 부트스트랩 학습된 모델 리스트
    model_type: ["MLP" | "DLinear" | "LSTM"]
    loader: inverse_transform 메소드를 가진 TimeSeriesDataLoader 객체
    반환: { ... }
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    def prep_inputs(X, y):
        if model_type == "MLP":
            return X.float().to(device).view(X.size(0), -1), y.float().to(device)
        return X.float().to(device), y.float().to(device)

    # [수정 1] 데이터로더에서 (X, y)만 받도록 수정
    def gather_targets(loader):
        ys = []
        for _, y in loader: # Unpacking 오류 수정
            ys.append(y)
        return torch.cat(ys, dim=0)

    # =================================================================
    # [수정 2] inverse 함수를 loader 객체를 사용하도록 전면 수정
    # =================================================================
    def inverse(tensor_data, data_loader_instance):
        # TimeSeriesDataLoader에 inverse_transform이 있는지 확인
        if hasattr(data_loader_instance, 'inverse_transform') and callable(data_loader_instance.inverse_transform):
            # 텐서를 CPU의 numpy 배열로 변환
            numpy_data = tensor_data.detach().cpu().numpy()
            # loader의 역변환 메소드 호출
            inversed_data = data_loader_instance.inverse_transform(numpy_data)
            return torch.from_numpy(inversed_data).float()
        else:
            # 역변환 메소드가 없으면 경고 출력 후 그대로 반환
            print("Warning: 'inverse_transform' method not found in loader. Returning scaled data.")
            return tensor_data

    # 1) 모든 실제 값(y) 수집
    Yv = gather_targets(valid_loader)  # [N_valid, L, d] (CPU)
    Yt = gather_targets(test_loader)   # [N_test,  L, d]

    # 2) 모든 모델의 예측값(y_hat)을 수집하고 평균
    Pv_list, Pt_list = [], []
    with torch.no_grad():
        for m in models:
            m.eval()
            m.to(device)

            # Validation set 예측
            outs_v = []
            for Xb, yb in valid_loader: # Unpacking 오류 수정
                Xb, _ = prep_inputs(Xb, yb)
                outs_v.append(m(Xb).detach().cpu())
            Pv_list.append(torch.cat(outs_v, dim=0))

            # Test set 예측
            outs_t = []
            for Xb, yb in test_loader: # Unpacking 오류 수정
                Xb, _ = prep_inputs(Xb, yb)
                outs_t.append(m(Xb).detach().cpu())
            Pt_list.append(torch.cat(outs_t, dim=0))

    # 부트스트랩 모델들의 예측을 평균
    Pv = torch.stack(Pv_list).mean(dim=0)  # [N_valid, L, d]
    Pt = torch.stack(Pt_list).mean(dim=0)  # [N_test,  L, d]

    # 3) 역변환 수행
    Yv_inv = inverse(Yv, loader)
    Pv_inv = inverse(Pv, loader)
    Yt_inv = inverse(Yt, loader)
    Pt_inv = inverse(Pt, loader)

    # 4) 잔차 계산
    Rv = Yv_inv - Pv_inv
    Rt = Yt_inv - Pt_inv

    return {
        "valid": {"y_true": Yv_inv, "y_pred": Pv_inv, "resid": Rv},
        "test":  {"y_true": Yt_inv, "y_pred": Pt_inv, "resid": Rt},
        "raw":   {
            "y_valid_scaled": Yv, "yhat_valid_scaled": Pv,
            "y_test_scaled":  Yt, "yhat_test_scaled":  Pt,
        },
    }