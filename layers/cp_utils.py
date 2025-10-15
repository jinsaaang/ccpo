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
import torch.optim as optim

def ellipsoid_volume(covariance_matrix, r):
    # Only compute volume of the ellipsoid along the first r dimensions
    # where r is the number of dimensions with non-zero eigenvalues
    eigenvalues = np.linalg.eigvals(covariance_matrix)
    eigenvalues = np.real(np.sort(eigenvalues)[::-1])
    eps = 1e-6
    num_r = np.sum(eigenvalues > eps)
    det_sigma = np.prod(eigenvalues[:num_r])
    constant_cd = np.pi**(num_r/2) / np.math.gamma(num_r/2 + 1)  # Volume constant for d-dimensional sphere
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


def train(model_cls, data_loader, valid_data_loader, EPOCHS=100, lr=1e-3, path='./weights/', patience=10, valid_mode=False):
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
            dlinear_configs = DLinearConfig(seq_len=seq_len, pred_len=pred_len, enc_in=enc_in)
            model_b = model_cls(dlinear_configs).to(device)
        else:
            raise ValueError(f"Unsupported model class: {model_cls}")
            
        optimizer = optim.Adam(model_b.parameters(), lr=lr)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        # ================================================================== #
        # 아래 부분을 수정하여 파일 이름에 모델 클래스 이름을 포함시킵니다.
        # 예: ./weights/MLP_model_b0.pt 또는 ./weights/DLinear_model_b0.pt
        model_name = model_cls.__name__ 
        model_save_path = f"{path}/{model_name}_model_b{i}.pt"
        # ================================================================== #

        for epoch in range(EPOCHS):
            model_b.train()
            total_train_loss = 0.0 
            for X_batch, y_batch, _, _ in loader_b:
                
                if model_cls == MLP:
                    X_batch = X_batch.float().to(device).view(X_batch.size(0), -1)
                    y_batch = y_batch.float().to(device).view(y_batch.size(0), -1)
                elif model_cls == DLinear:
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
                        elif model_cls == DLinear:
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


## sklearn에서 사용하는 형식을 따르는 PyTorch용 래퍼
class PyTorchRegressorWrapper:
    def __init__(self, model_class, model_params, optimizer_class, criterion, 
                 epochs=10, batch_size=32, lr=1e-3):
        # 모델 및 학습에 필요한 모든 구성요소를 저장
        self.model_class = model_class
        self.model_params = model_params
        self.optimizer_class = optimizer_class
        self.criterion = criterion
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.model = None # 학습이 시작되면 모델이 여기에 할당됩니다.

    def fit(self, X, y):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        """scikit-learn의 .fit()처럼 동작하는 메서드"""
        # 1. 모델과 옵티마이저 초기화
        self.model = self.model_class(**self.model_params).to(self.device)
        optimizer = self.optimizer_class(self.model.parameters(), lr=self.lr)

        # 2. 데이터로더 생성
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        dataset = TensorDataset(X_tensor, y_tensor)
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # 3. 모델 학습 루프
        self.model.train()
        for epoch in range(self.epochs):
            for x_batch, y_batch in data_loader:
                optimizer.zero_grad()
                outputs = self.model(x_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
        
        return self

    def predict(self, X):
        """scikit-learn의 .predict()처럼 동작하는 메서드"""
        if self.model is None:
            raise RuntimeError("You must call fit before predict")

        # 1. 모델을 평가 모드로 설정
        self.model.eval()
        
        # 2. 예측 수행
        X_tensor = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            predictions = self.model(X_tensor)
            
        # 3. 결과를 numpy 배열로 변환하여 반환
        return predictions.cpu().numpy()