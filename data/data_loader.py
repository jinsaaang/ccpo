import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from typing import Tuple, Optional, Literal, List
from pathlib import Path
from sklearn.preprocessing import StandardScaler

class TimeSeriesDataset(Dataset):
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class TimeSeriesDataLoader:
    
    def __init__(self, 
                 data_path: str = "snp50.csv",
                 base_path: str = "c:/Users/Kong/code/study_ccpo/data/"):
        self.data_path = Path(base_path) / data_path
        self.raw_data = None
        self.processed_data = None
        self.scaler = None
        
    def load_data(self) -> pd.DataFrame:
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        df = pd.read_csv(self.data_path, index_col=0, parse_dates=True)
        df = df.sort_index()
        
        self.raw_data = df
        print(f"Loaded data: {df.shape} ({df.index.min()} ~ {df.index.max()})")
        return df
    
    def preprocess_data(self) -> pd.DataFrame:
        if self.raw_data is None:
            self.load_data()
        
        data = self.raw_data.copy()
        
        # Forward fill missing values
        # data = data.ffill().bfill()
        # data = data.dropna()
        
        # Standard scaling
        self.scaler = StandardScaler()
        data_scaled = pd.DataFrame(
            self.scaler.fit_transform(data),
            index=data.index,
            columns=data.columns
        )
        
        self.processed_data = data_scaled
        print(f"Preprocessed data: {data_scaled.shape}")
        return data_scaled
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        if self.scaler is None:
            return data
        return self.scaler.inverse_transform(data)
    
    def resample_frequency(self, 
                          data: pd.DataFrame,
                          frequency: Literal['daily', 'weekly', 'monthly']) -> pd.DataFrame:
        if frequency == 'daily':
            return data
        elif frequency == 'weekly':
            resampled = data.resample('W-FRI').last()
        elif frequency == 'monthly':
            resampled = data.resample('M').last()
        else:
            raise ValueError("frequency must be 'daily', 'weekly', or 'monthly'")
        
        print(f"Resampled to {frequency}: {resampled.shape}")
        return resampled
    
    def create_sequences(self,
                        data: pd.DataFrame,
                        lookback: int,
                        forecast_horizon: int = 1) -> Tuple[np.ndarray, np.ndarray, List]:
        values = data.values
        dates = data.index
        
        X, y, pred_dates = [], [], []
        
        for i in range(lookback, len(values) - forecast_horizon + 1):
            X.append(values[i-lookback:i])
            y.append(values[i:i+forecast_horizon])
            pred_dates.append(dates[i+forecast_horizon-1])
        
        X = np.array(X)
        y = np.array(y)
        print(f"Created sequences - X: {X.shape}, y: {y.shape}")
        return X, y, pred_dates
    
    def create_sequences_KLV(self,
                            data: pd.DataFrame,
                            lookback: int,
                            K: int,
                            L: int, 
                            V: int,
                            start_idx: int = 0,
                            forecast_horizon: int = 1) -> Tuple:
        """
        Create sequences for K-L-V split with proper scaling
        
        Args:
            data: Raw price data (NOT scaled)
            lookback: History window size (LSTM input window, used within K/L/V data)
            K, L, V: Train/Calibration/Validation sizes (number of sequences/samples)
            start_idx: Starting index for rolling window
            forecast_horizon: Prediction horizon
            
        Returns:
            X_K, X_L, X_V: Input sequences
            y_K, y_L, y_V: Target values
            dates_K, dates_L, dates_V: Corresponding dates
            scaler: Fitted scaler (on training data only)
            
        Note:
            - lookback is LSTM input window, used WITHIN K/L/V data
            - K samples use the first K data points (with lookback from within K)
            - Total price data needed: K + L + V (lookback does NOT add extra data)
        """
        # Total price data points needed:
        # K, L, V only (lookback windows come from within these periods)
        total_needed = K + L + V
        end_idx = start_idx + total_needed
        
        if end_idx > len(data):
            raise ValueError(f"Not enough data: need {total_needed}, got {len(data) - start_idx}")
        
        # Extract window data
        window_data = data.iloc[start_idx:end_idx]
        
        # ⚠️ IMPORTANT: Convert price to returns for CCPO
        # CCPO predicts returns, not prices!
        returns_data = window_data.pct_change().dropna()
        
        # Adjust indices after dropna (lost 1 row)
        # We need to ensure we still have enough data
        if len(returns_data) < total_needed - 1:
            raise ValueError(f"Not enough data after returns conversion: need {total_needed-1}, got {len(returns_data)}")
        
        # Fit scaler on training data only (first K returns)
        # This ensures no data leakage from L or V sets
        train_cutoff = K - 1  # -1 because we lost 1 row to pct_change
        scaler = StandardScaler()
        scaler.fit(returns_data.iloc[:train_cutoff].values)
        
        # Transform entire window
        scaled_values = scaler.transform(returns_data.values)
        scaled_data = pd.DataFrame(
            scaled_values,
            index=returns_data.index,
            columns=returns_data.columns
        )
        
        # Create sequences
        # We create sequences starting from index 'lookback' within the data
        # This means first K-lookback sequences come from K period
        X, y, pred_dates = [], [], []
        for i in range(lookback, len(scaled_data) - forecast_horizon + 1):
            X.append(scaled_values[i-lookback:i])  # lookback-length input window
            y.append(scaled_values[i:i+forecast_horizon])  # forecast target
            pred_dates.append(scaled_data.index[i+forecast_horizon-1])
        
        X = np.array(X)
        y = np.array(y)
        if forecast_horizon == 1:
            y = y.squeeze(1)
        
        # Total sequences created: K + L + V - lookback
        # Split into K, L, V sequences
        # K gets first (K-lookback) sequences
        K_seq = K - lookback
        X_K = X[:K_seq]
        X_L = X[K_seq:K_seq+L]
        X_V = X[K_seq+L:K_seq+L+V]
        
        y_K = y[:K_seq]
        y_L = y[K_seq:K_seq+L]
        y_V = y[K_seq+L:K_seq+L+V]
        
        dates_K = pred_dates[:K_seq]
        dates_L = pred_dates[K_seq:K_seq+L]
        dates_V = pred_dates[K_seq+L:K_seq+L+V]
        
        return (X_K, X_L, X_V, y_K, y_L, y_V, 
                dates_K, dates_L, dates_V, scaler)
    
    def split_by_date(self,
                     X: np.ndarray,
                     y: np.ndarray,
                     dates: List,
                     train_end_date: str,
                     val_end_date: str,
                     test_end_date: Optional[str] = None) -> Tuple:
        dates = pd.to_datetime(dates)
        train_end = pd.to_datetime(train_end_date)
        val_end = pd.to_datetime(val_end_date)
        
        train_mask = dates <= train_end
        X_train = X[train_mask]
        y_train = y[train_mask]
        dates_train = dates[train_mask]
        
        val_mask = (dates > train_end) & (dates <= val_end)
        X_val = X[val_mask]
        y_val = y[val_mask]
        dates_val = dates[val_mask]
        
        if test_end_date is not None:
            test_end = pd.to_datetime(test_end_date)
            test_mask = (dates > val_end) & (dates <= test_end)
        else:
            test_mask = dates > val_end
        
        X_test = X[test_mask]
        y_test = y[test_mask]
        dates_test = dates[test_mask]
        
        print(f"\nData split by date:")
        print(f"Train: {len(X_train)} samples ({dates_train.min()} ~ {dates_train.max()})")
        print(f"Val:   {len(X_val)} samples ({dates_val.min()} ~ {dates_val.max()})")
        print(f"Test:  {len(X_test)} samples ({dates_test.min()} ~ {dates_test.max()})")
        
        return (X_train, X_val, X_test, 
                y_train, y_val, y_test,
                dates_train, dates_val, dates_test)
    
    def create_dataloaders(self,
                          frequency: Literal['daily', 'weekly', 'monthly'] = 'weekly',
                          lookback: int = 20,
                          forecast_horizon: int = 1,
                          train_end_date: str = '2020-12-31',
                          val_end_date: str = '2021-12-31',
                          test_end_date: Optional[str] = None,
                          batch_size: int = 32,
                          shuffle_train: bool = True,
                          use_scaler: bool = True) -> Tuple:
        self.load_data()

        if use_scaler:
            data_scaled = self.preprocess_data(train_end_date=train_end_date, 
                                       frequency=frequency)
        else:
            data_scaled = self.resample_frequency(data_scaled, frequency)
            self.scaler = None

        X, y, dates = self.create_sequences(
            data_scaled, 
            lookback=lookback,
            forecast_horizon=forecast_horizon
        )
        
        (X_train, X_val, X_test,
         y_train, y_val, y_test,
         dates_train, dates_val, dates_test) = self.split_by_date(
            X, y, dates,
            train_end_date=train_end_date,
            val_end_date=val_end_date,
            test_end_date=test_end_date
        )
        
        train_dataset = TimeSeriesDataset(X_train, y_train)
        val_dataset = TimeSeriesDataset(X_val, y_val)
        test_dataset = TimeSeriesDataset(X_test, y_test)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=shuffle_train
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False
        )
        
        dates_dict = {
            'train': dates_train,
            'val': dates_val,
            'test': dates_test
        }
        
        print(f"\nDataLoaders created successfully!")
        print(f"Batch size: {batch_size}")
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        print(f"Test batches: {len(test_loader)}")
        
        return train_loader, val_loader, test_loader, dates_dict, self.scaler
    

# Usage example
# loader = TimeSeriesDataLoader()

# # Create dataloaders with weekly data
# train_loader, val_loader, test_loader, dates_dict, scaler = loader.create_dataloaders(
#     frequency='weekly',
#     lookback=12,  # 12 weeks of history
#     forecast_horizon=1,  # Predict 1 week ahead
#     train_end_date='2020-12-31',
#     val_end_date='2021-12-31',
#     scale_method='standard',
#     batch_size=32
# )

# # Check data
# for batch_X, batch_y in train_loader:
#     print(f"\nBatch X shape: {batch_X.shape}")  # [batch, lookback, features]
#     print(f"Batch y shape: {batch_y.shape}")    # [batch, features]
#     break

# # Inverse transform example
# if scaler is not None:
#     y_original = loader.inverse_transform(batch_y.numpy())
#     print(f"Inverse transformed y shape: {y_original.shape}")