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
        
        if forecast_horizon == 1:
            y = y.squeeze(1)
        
        print(f"Created sequences - X: {X.shape}, y: {y.shape}")
        return X, y, pred_dates
    
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
                          shuffle_train: bool = True) -> Tuple:
        
        self.preprocess_data()
        
        data_resampled = self.resample_frequency(self.processed_data, frequency)
        
        X, y, dates = self.create_sequences(
            data_resampled, 
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