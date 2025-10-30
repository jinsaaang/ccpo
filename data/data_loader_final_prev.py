import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from typing import Tuple, Optional, List, Union, Dict, Literal
from pathlib import Path
from sklearn.preprocessing import StandardScaler


# -----------------------
# Dataset (for model loaders)
# -----------------------
class TimeSeriesDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, unsqueeze_y: bool = False):
        self.X = torch.FloatTensor(X)
        y_t = torch.FloatTensor(y)
        if unsqueeze_y and y_t.ndim == 2:
            y_t = y_t.unsqueeze(1)  # [N, d] -> [N, 1, d]
        self.y = y_t

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# -----------------------
# Main Loader
# -----------------------
class TimeSeriesDataLoader:
    def __init__(
        self,
        base_path: str = "./data/",
        num_assets: int = 10,
    ):
        # <--- MODIFIED SECTION START --->
        def get_data_path(num_assets: int, base_dir: Union[str, Path] = ".") -> Path:
            allowed_assets = {5, 10, 30, 49}
            if num_assets not in allowed_assets:
                raise ValueError(f"num_assets must be one of {sorted(allowed_assets)}")
            
            # data_type을 'daily'로 고정
            return Path(base_dir) / f"industry_{num_assets}_daily.csv"

        self.data_path = get_data_path(num_assets, base_path)
        # <--- MODIFIED SECTION END --->
        
        self.raw_data: Optional[pd.DataFrame] = None
        self.scaler: Optional[StandardScaler] = None  # fit ONLY on "model train"

    # -------- I/O --------
    def load_data(self) -> pd.DataFrame:
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        df = pd.read_csv(self.data_path, index_col=0, parse_dates=True).sort_index()
        self.raw_data = df
        print(f"Loaded data: {df.shape} ({df.index.min()} ~ {df.index.max()})")
        return df

    # -------- Resample Helper --------
    def resample_frequency(
        self, 
        data: pd.DataFrame,
        frequency: Literal['daily', 'weekly']
    ) -> pd.DataFrame:
        
        if frequency == 'daily':
            print("Using original 'daily' data.")
            return data
        
        elif frequency == 'weekly':
            # 'W-FRI' (금요일 기준)으로 리샘플링하고, 빈 값(NaN)은 제거
            resampled = data.resample('W-FRI').last().dropna()
        
        else:
            raise ValueError("frequency must be 'daily' or 'weekly'")
        
        if len(resampled) == 0:
             raise ValueError(f"Resampling to {frequency} ('W-FRI') resulted in empty DataFrame.")

        print(f"Resampled data to {frequency} ('W-FRI'): {resampled.shape}")
        return resampled

    # -------- Scaling (fit/transform 분리) --------
    def fit_scaler(self, fit_df: pd.DataFrame):
        """스케일러는 반드시 '모델 Train(=K 시작 이전)' 구간으로만 fit."""
        self.scaler = StandardScaler().fit(fit_df.values)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.scaler is None:
            return df.copy()
        arr = self.scaler.transform(df.values)
        return pd.DataFrame(arr, index=df.index, columns=df.columns)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        
        if self.scaler is None:
            return data  # 스케일러 미사용 시 패스

        if data.ndim == 1:
            out = self.scaler.inverse_transform(data.reshape(1, -1))
            return out.reshape(-1)  # (features,)
        elif data.ndim == 2:
            return self.scaler.inverse_transform(data)
        elif data.ndim == 3:
            n, h, f = data.shape
            out2d = self.scaler.inverse_transform(data.reshape(n * h, f))
            return out2d.reshape(n, h, f)
        else:
            raise ValueError(f"inverse_transform: unsupported ndim={data.ndim}")

    # -------- Sequence 생성 (forecast_horizon=1 고정) --------
    def create_sequences(
        self,
        data: pd.DataFrame,
        lookback: int,
    ) -> Tuple[np.ndarray, np.ndarray, List[pd.Timestamp]]:
        """
        horizon=1 고정.
        X: [N, lookback, d], y: [N, d]
        pred_dates: 각 y 시점의 날짜
        """
        values = data.values
        dates = data.index
        X, y, pred_dates = [], [], []
        for i in range(lookback, len(values)):
            X.append(values[i - lookback : i])
            y.append(values[i])  # horizon=1
            pred_dates.append(dates[i])
        X = np.array(X)
        y = np.array(y)
        print(f"Created sequences - X: {X.shape}, y: {y.shape}")
        return X, y, pred_dates

    # =========================
    # 날짜 기준: Model(train/valid/test) + Opt(y_K/y_V only, raw scale)
    # =========================
    def create_all_by_dates(
        self,
        lookback: int,
        train_end_date: str,      # 모델 Train 마지막 날짜
        val_end_date: str,        # 모델 Valid(=K) 마지막 날짜
        test_end_date: Optional[str] = None,  # 모델 Test(=V) 마지막 날짜 (None이면 끝까지)
        batch_size: int = 32,
        shuffle_train: bool = True,
        use_scaler: bool = True,
        frequency: Literal['daily', 'weekly'] = 'daily',
    ) -> Dict[str, object]:
        if self.raw_data is None:
            self.load_data()

        data_to_use = self.resample_frequency(self.raw_data, frequency)

        train_end = pd.to_datetime(train_end_date)
        val_end   = pd.to_datetime(val_end_date)

        # 1) scaler fit on model-train only (K 시작 전)
        if use_scaler:
            fit_df = data_to_use.loc[:train_end]
            self.fit_scaler(fit_df)

        # 2) 모델용: 스케일 데이터로 시퀀스
        data_scaled = self.transform(data_to_use)
        X_s, y_s, dates = self.create_sequences(data_scaled, lookback=lookback)
        dates = pd.to_datetime(dates)

        # 3) 최적화용: 원본(raw) 데이터로 시퀀스 (스케일 X)
        _, y_raw, dates_raw = self.create_sequences(data_to_use, lookback)
        assert len(dates) == len(dates_raw) and np.all(dates == pd.to_datetime(dates_raw)), \
            "Scaled/Raw sequence dates misaligned."

        # 4) masks
        train_mask = dates <= train_end
        valid_mask = (dates > train_end) & (dates <= val_end)
        if test_end_date is not None:
            test_end = pd.to_datetime(test_end_date)
            test_mask = (dates > val_end) & (dates <= test_end)
        else:
            test_mask = dates > val_end

        # 5) 모델용(스케일) 분할 → DataLoader(+unsqueeze_y)
        X_tr, y_tr, d_tr = X_s[train_mask], y_s[train_mask], dates[train_mask]
        X_va, y_va, d_va = X_s[valid_mask], y_s[valid_mask], dates[valid_mask]  # == K
        X_te, y_te, d_te = X_s[test_mask],  y_s[test_mask],  dates[test_mask]   # == V

        model_train_loader = DataLoader(TimeSeriesDataset(X_tr, y_tr, unsqueeze_y=True), batch_size=batch_size, shuffle=shuffle_train)
        model_valid_loader = DataLoader(TimeSeriesDataset(X_va, y_va, unsqueeze_y=True), batch_size=batch_size, shuffle=False)
        model_test_loader  = DataLoader(TimeSeriesDataset(X_te, y_te, unsqueeze_y=True), batch_size=batch_size, shuffle=False)

        # 6) 최적화용(원본) → **y_K, y_V만** 추출
        y_K = y_raw[valid_mask]
        y_V = y_raw[test_mask]
        t_K = np.array(dates[valid_mask])
        t_V = np.array(dates[test_mask])

        print("\n[Date mode] Model Split")
        print(f"Train: {len(X_tr)} ({d_tr.min() if len(d_tr)>0 else None} ~ {d_tr.max() if len(d_tr)>0 else None})")
        print(f"Valid(K): {len(X_va)} ({d_va.min() if len(d_va)>0 else None} ~ {d_va.max() if len(d_va)>0 else None})")
        print(f"Test (V): {len(X_te)} ({d_te.min() if len(d_te)>0 else None} ~ {d_te.max() if len(d_te)>0 else None})")

        return {
            "model": {
                "train_loader": model_train_loader,
                "valid_loader": model_valid_loader,
                "test_loader":  model_test_loader,
                "dates": {"train": np.array(d_tr), "valid": np.array(d_va), "test": np.array(d_te)},
            },
            "opt": {
                "y_K": y_K, "dates_K": t_K,
                "y_V": y_V, "dates_V": t_V,
            },
            "scaler": self.scaler,
        }

    # =========================
    # 개수 기준: Model(train_len / K / V) + Opt(y_K/y_V only, raw scale)
    # =========================
    def create_all_by_counts(
        self,
        lookback: int,
        train_len: int,  # 모델 Train 시퀀스 개수(= K 시작 이전)
        K: int,          # Valid(K) 시퀀스 개수
        V: int,          # Test(V) 시퀀스 개수
        start_idx: int = 0,
        batch_size: int = 32,
        shuffle_train: bool = True,
        use_scaler: bool = True,
        frequency: Literal['daily', 'weekly'] = 'daily',
    ) -> Dict[str, object]:
        if self.raw_data is None:
            self.load_data()

        data_to_use = self.resample_frequency(self.raw_data, frequency)

        total_needed = lookback + train_len + K + V
        end_idx = start_idx + total_needed
        
        if end_idx > len(data_to_use):
            raise ValueError(
                f"Not enough data: need {total_needed}, available {len(data_to_use) - start_idx} (after resampling)"
            )

        # 1) window
        window_df = data_to_use.iloc[start_idx:end_idx]

        # 2) scaler fit on model-train only
        if use_scaler:
            fit_df = window_df.iloc[: (lookback + train_len)]
            self.fit_scaler(fit_df)

        # 3) 모델용(스케일) 시퀀스
        scaled_df = self.transform(window_df)
        X_all, y_all, dates_all = self.create_sequences(scaled_df, lookback=lookback)

        # 4) 최적화용(원본) 시퀀스
        _, yr, dtr = self.create_sequences(window_df, lookback)
        assert len(dates_all) == len(dtr), "Scaled/Raw sequence length mismatch in count mode."

        required = train_len + K + V
        if len(X_all) < required:
            raise ValueError(
                f"Sequence count {len(X_all)} is less than required train_len+K+V={required}."
            )

        # 5) 인덱스 스플릿
        tr_slice = slice(0, train_len)
        va_slice = slice(train_len, train_len + K)
        te_slice = slice(train_len + K, train_len + K + V)

        # 모델용(스케일) → DataLoader(+unsqueeze_y)
        X_tr, y_tr = X_all[tr_slice], y_all[tr_slice]
        X_va, y_va = X_all[va_slice], y_all[va_slice]  # == K
        X_te, y_te = X_all[te_slice], y_all[te_slice]  # == V

        d_tr = np.array(dates_all[tr_slice])
        d_va = np.array(dates_all[va_slice])
        d_te = np.array(dates_all[te_slice])

        model_train_loader = DataLoader(TimeSeriesDataset(X_tr, y_tr, unsqueeze_y=True), batch_size=batch_size, shuffle=shuffle_train)
        model_valid_loader = DataLoader(TimeSeriesDataset(X_va, y_va, unsqueeze_y=True), batch_size=batch_size, shuffle=False)
        model_test_loader  = DataLoader(TimeSeriesDataset(X_te, y_te, unsqueeze_y=True), batch_size=batch_size, shuffle=False)

        # 최적화용(원본) → **y_K, y_V만** 추출
        y_K = yr[va_slice]
        y_V = yr[te_slice]
        t_K = np.array(dtr[va_slice])
        t_V = np.array(dtr[te_slice])

        print("\n[Count mode] Model Split")
        print(f"Start idx: {start_idx}, lookback: {lookback}, train_len={train_len}, K={K}, V={V}")
        print(f"Train: {len(X_tr)} ({d_tr[0] if len(d_tr)>0 else None} ~ {d_tr[-1] if len(d_tr)>0 else None})")
        print(f"Valid(K): {len(X_va)} ({d_va[0] if len(d_va)>0 else None} ~ {d_va[-1] if len(d_va)>0 else None})")
        print(f"Test (V): {len(X_te)} ({d_te[0] if len(d_te)>0 else None} ~ {d_te[-1] if len(d_te)>0 else None})")

        return {
            "model": {
                "train_loader": model_train_loader,
                "valid_loader": model_valid_loader,
                "test_loader":  model_test_loader,
                "dates": {"train": d_tr, "valid": d_va, "test": d_te},
            },
            "opt": {
                "y_K": y_K, "dates_K": t_K,
                "y_V": y_V, "dates_V": t_V,
            },
            "scaler": self.scaler,
        }

    # =========================
    # 별도: K-L-V 시퀀스 생성기 (배열 반환, L=0 허용)
    # =========================
    def create_sequences_KLV(
        self,
        data: pd.DataFrame,   # RAW (미스케일)
        lookback: int,
        K: int,
        L: int,
        V: int,
        start_idx: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
               np.ndarray, np.ndarray, np.ndarray,
               np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
        """
        Data Construction for K-L-V (horizon=1, L=0 허용)
        """
        total_needed = lookback + K + L + V
        end_idx = start_idx + total_needed
        if end_idx > len(data):
            raise ValueError(f"Not enough data: need {total_needed}, got {len(data) - start_idx}")

        window_data = data.iloc[start_idx:end_idx]

        # fit scaler on lookback+K (train only)
        train_cutoff = lookback + K
        scaler = StandardScaler()
        scaler.fit(window_data.iloc[:train_cutoff].values)

        # transform
        scaled_values = scaler.transform(window_data.values)
        scaled_data = pd.DataFrame(scaled_values, index=window_data.index, columns=window_data.columns)

        # sequences (horizon=1)
        X, y, pred_dates = [], [], []
        for i in range(lookback, len(scaled_data)):
            X.append(scaled_values[i - lookback : i])
            y.append(scaled_values[i])
            pred_dates.append(scaled_data.index[i])

        X = np.array(X)
        y = np.array(y)

        # split K/L/V (L=0 허용)
        X_K = X[:K]
        X_L = X[K:K+L] if L > 0 else np.empty((0,) + X.shape[1:], dtype=X.dtype)
        X_V = X[K+L:K+L+V]

        y_K = y[:K]
        y_L = y[K:K+L] if L > 0 else np.empty((0,) + y.shape[1:], dtype=y.dtype)
        y_V = y[K+L:K+L+V]

        dates_K = np.array(pred_dates[:K])
        dates_L = np.array(pred_dates[K:K+L]) if L > 0 else np.empty((0,), dtype=object)
        dates_V = np.array(pred_dates[K+L:K+L+V])

        return (X_K, X_L, X_V, y_K, y_L, y_V, dates_K, dates_L, dates_V, scaler)

    
    def _require_keys(self, kwargs: Dict, keys: List[str]):
        """Helper to check for required keys in kwargs."""
        missing = [k for k in keys if k not in kwargs]
        if missing:
            raise ValueError(f"Missing required arguments for mode: {missing}")
    
    
    # =========================
    # Wrapper
    # =========================
    def create_all(
        self,
        mode: Literal["counts", "dates"],
        **kwargs
    ):
        """
        mode에 따라 적절한 스플릿 함수를 호출한다.
        
        [resample_freq] (kwargs로 전달)
        - 'daily' (기본값) : 리샘플링 안함 (원본 데이터 사용)
        - 'weekly'        : 주간 리샘플링 ('W-FRI' - 금요일 기준)
        
        - mode='counts' -> create_all_by_counts(...)
            필수: lookback, train_len, K, V
            선택: start_idx, batch_size, shuffle_train, use_scaler, resample_freq
        - mode='dates'  -> create_all_by_dates(...)
            필수: lookback, train_end_date, val_end_date
            선택: test_end_date, batch_size, shuffle_train, use_scaler, resample_freq
        """
        
        # 1. 'resample_freq' 인자 추출 (기본값 'daily')
        frequency = kwargs.get("resample_freq", 'daily')
        
        if frequency not in ['daily', 'weekly']:
            raise ValueError(f"resample_freq must be 'daily' or 'weekly', but got {frequency}")
            
        
        # 2. 모드에 따라 하위 함수 호출
        if mode == "counts":
            self._require_keys(kwargs, ["lookback", "train_len", "K", "V"])
            return self.create_all_by_counts(
                lookback=kwargs["lookback"],
                train_len=kwargs["train_len"],
                K=kwargs["K"],
                V=kwargs["V"],
                start_idx=kwargs.get("start_idx", 0),
                batch_size=kwargs.get("batch_size", 32),
                shuffle_train=kwargs.get("shuffle_train", True),
                use_scaler=kwargs.get("use_scaler", True),
                frequency=frequency,
            )
        elif mode == "dates":
            self._require_keys(kwargs, ["lookback", "train_end_date", "val_end_date"])
            return self.create_all_by_dates(
                lookback=kwargs["lookback"],
                train_end_date=kwargs["train_end_date"],
                val_end_date=kwargs["val_end_date"],
                test_end_date=kwargs.get("test_end_date", None),
                batch_size=kwargs.get("batch_size", 32),
                shuffle_train=kwargs.get("shuffle_train", True),
                use_scaler=kwargs.get("use_scaler", True),
                frequency=frequency,
            )
        else:
            raise ValueError("mode must be 'counts' or 'dates'")