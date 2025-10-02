# preprocess -> dataloader (daily, weekly, monthly)

import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from typing import Tuple, Optional, Literal
from pathlib import Path
import warnings

class FactorDataset(Dataset):
    """Factor와 수익률 데이터를 위한 Dataset 클래스"""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class FactorDataLoader:
    """
    Factor 데이터와 수익률 데이터를 로드하고 전처리하는 클래스
    """
    
    def __init__(self, 
                 factor_path: str = "factors.csv",
                 industry_path: str = "industry49.csv",
                 base_path: str = "c:/Users/Kong/code/study_ccpo/data/"):
        self.factor_path = Path(base_path) / factor_path
        self.industry_path = Path(base_path) / industry_path
        self.factor_data = None
        self.industry_data = None
        
    def load_factor_data(self) -> pd.DataFrame:
        """Factor 데이터 로드"""
        if not self.factor_path.exists():
            raise FileNotFoundError(f"Factor file not found: {self.factor_path}")
        
        # 단일 factor 파일 로드
        df = pd.read_csv(self.factor_path)
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
        elif 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
        
        self.factor_data = df.sort_index()
        
        print(f"Loaded factor data: {self.factor_data.shape}")
        return self.factor_data
    
    def load_industry_data(self) -> pd.DataFrame:
        """Industry 데이터 로드"""
        self.industry_data = pd.read_csv(self.industry_path)
        
        if 'Date' in self.industry_data.columns:
            self.industry_data['Date'] = pd.to_datetime(self.industry_data['Date'])
            self.industry_data.set_index('Date', inplace=True)
        elif 'date' in self.industry_data.columns:
            self.industry_data['date'] = pd.to_datetime(self.industry_data['date'])
            self.industry_data.set_index('date', inplace=True)
        
        self.industry_data = self.industry_data.sort_index()
        
        print(f"Loaded industry data: {self.industry_data.shape}")
        return self.industry_data
    
    def preprocess_data(self, 
                       fillna_method: str = 'forward',
                       normalize: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """데이터 전처리"""
        if self.factor_data is None:
            self.load_factor_data()
        if self.industry_data is None:
            self.load_industry_data()
        
        # 결측치 처리
        if fillna_method == 'forward':
            self.factor_data = self.factor_data.ffill()
            self.industry_data = self.industry_data.ffill()
        elif fillna_method == 'backward':
            self.factor_data = self.factor_data.bfill()
            self.industry_data = self.industry_data.bfill()
        elif fillna_method == 'mean':
            self.factor_data = self.factor_data.fillna(self.factor_data.mean())
            self.industry_data = self.industry_data.fillna(self.industry_data.mean())
        
        # 정규화
        if normalize:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            factor_cols = self.factor_data.columns
            self.factor_data[factor_cols] = scaler.fit_transform(self.factor_data[factor_cols])
        
        return self.factor_data, self.industry_data
    
    def create_frequency_data(self, 
                            frequency: Literal['daily', 'weekly', 'monthly'] = 'weekly',
                            factor_timing: Literal['start', 'end'] = 'start') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        주어진 빈도로 데이터 변환
        
        Args:
            frequency: 'daily', 'weekly' 또는 'monthly'
            factor_timing: 'start' (기간 시작일 팩터 사용) 또는 'end' (기간 마지막일 팩터 사용)
        """
        if self.factor_data is None or self.industry_data is None:
            self.preprocess_data()
        
        # 공통 날짜 범위 찾기
        common_start = max(self.factor_data.index.min(), self.industry_data.index.min())
        common_end = min(self.factor_data.index.max(), self.industry_data.index.max())
        
        factor_data_common = self.factor_data[common_start:common_end]
        industry_data_common = self.industry_data[common_start:common_end]
        
        if frequency == 'daily':
            return self._create_daily_data(factor_data_common, industry_data_common, factor_timing)
        elif frequency == 'weekly':
            return self._create_weekly_data(factor_data_common, industry_data_common, factor_timing)
        elif frequency == 'monthly':
            return self._create_monthly_data(factor_data_common, industry_data_common, factor_timing)
        else:
            raise ValueError("frequency must be 'daily', 'weekly' or 'monthly'")
    
    def _create_weekly_data(self, 
                          factor_data: pd.DataFrame, 
                          industry_data: pd.DataFrame,
                          factor_timing: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """주간 데이터 생성"""
        
        # 원본 데이터를 복사하여 수정
        factor_copy = factor_data.copy()
        industry_copy = industry_data.copy()
        
        # 주간 그룹 생성 (월요일 시작)
        factor_copy['week'] = factor_copy.index.to_series().dt.to_period('W-MON')
        industry_copy['week'] = industry_copy.index.to_series().dt.to_period('W-MON')
        
        # Factor 데이터 처리
        if factor_timing == 'start':
            # 각 주의 첫 거래일 (월요일 또는 그 이후 첫 거래일)
            weekly_factors = factor_copy.groupby('week').first()
        else:  # factor_timing == 'end'
            # 각 주의 마지막 거래일 (금요일 또는 그 이전 마지막 거래일)
            weekly_factors = factor_copy.groupby('week').last()
        
        # 수익률 계산 (주간 마지막 거래일 기준)
        weekly_prices = industry_copy.groupby('week').last()
        
        # 수익률 계산: (현재주 마지막일 가격 - 이전주 마지막일 가격) / 이전주 마지막일 가격
        weekly_returns = weekly_prices.pct_change().dropna()
        
        # week 컬럼 제거 (groupby 후에는 week가 인덱스가 아닌 컬럼으로 남아있을 수 있음)
        if 'week' in weekly_factors.columns:
            weekly_factors = weekly_factors.drop('week', axis=1)
        if 'week' in weekly_returns.columns:
            weekly_returns = weekly_returns.drop('week', axis=1)
        
        # 시간 정렬 확인
        common_weeks = weekly_factors.index.intersection(weekly_returns.index)
        weekly_factors = weekly_factors.loc[common_weeks]
        weekly_returns = weekly_returns.loc[common_weeks]
        
        # Period 인덱스를 datetime으로 변환 (각 주의 시작일로)
        weekly_factors.index = weekly_factors.index.to_timestamp()
        weekly_returns.index = weekly_returns.index.to_timestamp()
        
        print(f"Created weekly data - Factors: {weekly_factors.shape}, Returns: {weekly_returns.shape}")
        return weekly_factors, weekly_returns
    
    def _create_daily_data(self, 
                         factor_data: pd.DataFrame, 
                         industry_data: pd.DataFrame,
                         factor_timing: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """일간 데이터 생성"""
        
        # Factor 데이터 처리
        if factor_timing == 'start':
            # 당일 시작 팩터 사용
            daily_factors = factor_data.copy()
        else:  # factor_timing == 'end'
            # 전일 마지막 팩터 사용 (1일 shift)
            daily_factors = factor_data.shift(1).dropna()
        
        # 수익률 계산: (당일 종가 - 전일 종가) / 전일 종가
        daily_returns = industry_data.pct_change().dropna()
        
        # 공통 날짜로 정렬
        common_dates = daily_factors.index.intersection(daily_returns.index)
        daily_factors = daily_factors.loc[common_dates]
        daily_returns = daily_returns.loc[common_dates]
        
        print(f"Created daily data - Factors: {daily_factors.shape}, Returns: {daily_returns.shape}")
        return daily_factors, daily_returns
    
    def _create_monthly_data(self, 
                           factor_data: pd.DataFrame, 
                           industry_data: pd.DataFrame,
                           factor_timing: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """월간 데이터 생성"""
        
        # 원본 데이터를 복사하여 수정
        factor_copy = factor_data.copy()
        industry_copy = industry_data.copy()
        
        # 월간 그룹 생성
        factor_copy['month'] = factor_copy.index.to_series().dt.to_period('M')
        industry_copy['month'] = industry_copy.index.to_series().dt.to_period('M')
        
        # Factor 데이터 처리
        if factor_timing == 'start':
            # 각 월의 첫 거래일
            monthly_factors = factor_copy.groupby('month').first()
        else:  # factor_timing == 'end'
            # 각 월의 마지막 거래일
            monthly_factors = factor_copy.groupby('month').last()
        
        # 수익률 계산 (월간 마지막 거래일 기준)
        monthly_prices = industry_copy.groupby('month').last()
        
        # 수익률 계산
        monthly_returns = monthly_prices.pct_change().dropna()
        
        # month 컬럼 제거
        if 'month' in monthly_factors.columns:
            monthly_factors = monthly_factors.drop('month', axis=1)
        if 'month' in monthly_returns.columns:
            monthly_returns = monthly_returns.drop('month', axis=1)
        
        # 시간 정렬 확인
        common_months = monthly_factors.index.intersection(monthly_returns.index)
        monthly_factors = monthly_factors.loc[common_months]
        monthly_returns = monthly_returns.loc[common_months]
        
        # Period 인덱스를 datetime으로 변환 (각 월의 시작일로)
        monthly_factors.index = monthly_factors.index.to_timestamp()
        monthly_returns.index = monthly_returns.index.to_timestamp()
        
        print(f"Created monthly data - Factors: {monthly_factors.shape}, Returns: {monthly_returns.shape}")
        return monthly_factors, monthly_returns
    
    def train_test_split(self, 
                        X: pd.DataFrame, 
                        y: pd.DataFrame,
                        train_end_date: Optional[str] = None,
                        val_end_date: Optional[str] = None,
                        test_end_date: Optional[str] = None,
                        test_ratio: float = 0.2,
                        val_ratio: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        시계열 데이터 분할
        
        Args:
            X, y: 입력 데이터
            train_end_date: 훈련 데이터 종료일 (YYYY-MM-DD 형식)
            val_end_date: 검증 데이터 종료일 (YYYY-MM-DD 형식)
            test_ratio, val_ratio: 날짜 기반 분할을 사용하지 않을 때 비율
        """
        
        # 날짜 기반 분할
        if train_end_date is not None:
            train_end_date = pd.to_datetime(train_end_date)
            
            # 훈련 데이터
            X_train = X[X.index <= train_end_date]
            y_train = y[y.index <= train_end_date]
            
            if val_end_date is not None:
                val_end_date = pd.to_datetime(val_end_date)
                
                # 검증 데이터
                X_val = X[(X.index > train_end_date) & (X.index <= val_end_date)]
                y_val = y[(y.index > train_end_date) & (y.index <= val_end_date)]
                
                # 테스트 데이터
                X_test = X[X.index > val_end_date]
                y_test = y[y.index > val_end_date]
            else:
                # val_end_date가 없으면 나머지를 반반 분할
                remaining_X = X[X.index > train_end_date]
                remaining_y = y[y.index > train_end_date]
                
                n_remaining = len(remaining_X)
                n_val = n_remaining // 2
                
                X_val = remaining_X.iloc[:n_val]
                X_test = remaining_X.iloc[n_val:]
                y_val = remaining_y.iloc[:n_val]
                y_test = remaining_y.iloc[n_val:]
        
        # 비율 기반 분할 (기존 방식)
        else:
            n_total = len(X)
            n_test = int(n_total * test_ratio)
            n_val = int(n_total * val_ratio)
            n_train = n_total - n_test - n_val
            
            # 시계열 순서 유지하여 분할
            X_train = X.iloc[:n_train]
            X_val = X.iloc[n_train:n_train+n_val]
            X_test = X.iloc[n_train+n_val:]
            
            y_train = y.iloc[:n_train]
            y_val = y.iloc[n_train:n_train+n_val]
            y_test = y.iloc[n_train+n_val:]
        
        print(f"Data split - Train: {len(X_train)} ({X_train.index.min()} ~ {X_train.index.max()})")
        print(f"           - Val: {len(X_val)} ({X_val.index.min()} ~ {X_val.index.max()})")
        print(f"           - Test: {len(X_test)} ({X_test.index.min()} ~ {X_test.index.max()})")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def create_dataloaders(self,
                          frequency: Literal['daily', 'weekly', 'monthly'] = 'weekly',
                          factor_timing: Literal['start', 'end'] = 'start',
                          train_end_date: Optional[str] = None,
                          val_end_date: Optional[str] = None,
                          test_end_date: Optional[str] = None,
                          test_ratio: float = 0.2,
                          val_ratio: float = 0.1,
                          batch_size: int = 32,
                          shuffle_train: bool = True) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """전체 파이프라인을 통해 DataLoader 생성"""
        
        # 데이터 로드 및 전처리
        self.preprocess_data()
        
        # 빈도별 데이터 생성
        X, y = self.create_frequency_data(frequency, factor_timing)
        
        # 데이터 분할
        X_train, X_val, X_test, y_train, y_val, y_test = self.train_test_split(
            X, y, train_end_date, val_end_date, test_end_date, test_ratio, val_ratio
        )
        
        # Dataset 생성
        train_dataset = FactorDataset(X_train.values, y_train.values)
        val_dataset = FactorDataset(X_val.values, y_val.values)
        test_dataset = FactorDataset(X_test.values, y_test.values)
        
        # DataLoader 생성
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader

# 사용 예시
if __name__ == "__main__":
    # 데이터 로더 생성
    loader = FactorDataLoader()
    
    # 주간 데이터로 DataLoader 생성 (팩터는 주 시작일 사용)
    train_loader, val_loader, test_loader = loader.create_dataloaders(
        frequency='weekly',
        factor_timing='start',  # 주 시작일(월요일) 팩터 사용
        train_end_date='2020-12-31',  # 2021년 이전까지 훈련
        val_end_date='2021-12-31',    # 2022년 이전까지 검증
        batch_size=64
    )
    
    # 데이터 확인
    for batch_X, batch_y in train_loader:
        print(f"Batch X shape: {batch_X.shape}")
        print(f"Batch y shape: {batch_y.shape}")
        break