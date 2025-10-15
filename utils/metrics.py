import numpy as np
import pandas as pd
from typing import Dict, List

def calculate_cumulative_return(returns: np.ndarray) -> float:
    """Calculate cumulative return"""
    return np.prod(1 + returns) - 1

def calculate_annualized_return(returns: np.ndarray, periods_per_year: int = 252) -> float:
    """Calculate annualized return"""
    cum_return = calculate_cumulative_return(returns)
    n_periods = len(returns)
    return (1 + cum_return) ** (periods_per_year / n_periods) - 1

def calculate_volatility(returns: np.ndarray, periods_per_year: int = 252) -> float:
    """Calculate annualized volatility"""
    return np.std(returns, ddof=1) * np.sqrt(periods_per_year)

def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0, 
                          periods_per_year: int = 252) -> float:
    """Calculate Sharpe ratio"""
    excess_returns = returns - risk_free_rate / periods_per_year
    if np.std(excess_returns, ddof=1) == 0:
        return 0.0
    return np.mean(excess_returns) / np.std(excess_returns, ddof=1) * np.sqrt(periods_per_year)

def calculate_max_drawdown(returns: np.ndarray) -> float:
    """Calculate maximum drawdown"""
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    return np.min(drawdown)

def calculate_turnover(weights_history: List[np.ndarray]) -> float:
    """Calculate average portfolio turnover"""
    if len(weights_history) < 2:
        return 0.0
    
    turnovers = []
    for i in range(1, len(weights_history)):
        turnover = np.sum(np.abs(weights_history[i] - weights_history[i-1]))
        turnovers.append(turnover)
    
    return np.mean(turnovers)

def calculate_violation_rate(portfolio_returns: np.ndarray, 
                            threshold: float) -> float:
    """Calculate violation rate (coverage rate)"""
    violations = np.sum(portfolio_returns < threshold)
    return violations / len(portfolio_returns)

def calculate_all_metrics(portfolio_returns: np.ndarray,
                         weights_history: List[np.ndarray],
                         threshold: float = 0.0,
                         solve_times: List[float] = None,
                         risk_free_rate: float = 0.0,
                         periods_per_year: int = 252) -> Dict:
    """Calculate all portfolio metrics"""
    metrics = {}
    
    metrics['cumulative_return'] = calculate_cumulative_return(portfolio_returns)
    metrics['annualized_return'] = calculate_annualized_return(portfolio_returns, periods_per_year)
    metrics['volatility'] = calculate_volatility(portfolio_returns, periods_per_year)
    metrics['sharpe_ratio'] = calculate_sharpe_ratio(portfolio_returns, risk_free_rate, periods_per_year)
    metrics['max_drawdown'] = calculate_max_drawdown(portfolio_returns)
    metrics['violation_rate'] = calculate_violation_rate(portfolio_returns, threshold)
    metrics['coverage_rate'] = 1 - metrics['violation_rate']
    metrics['turnover'] = calculate_turnover(weights_history)
    
    if solve_times:
        metrics['avg_solve_time'] = np.mean(solve_times)
        metrics['total_solve_time'] = np.sum(solve_times)
    else:
        metrics['avg_solve_time'] = 0.0
        metrics['total_solve_time'] = 0.0
    
    return metrics

def print_metrics(metrics: Dict, portfolio_name: str = "Portfolio"):
    """Print metrics in a formatted way"""
    print(f"\n{'='*60}")
    print(f"{portfolio_name} Performance Metrics")
    print(f"{'='*60}")
    
    print(f"\n📊 Return Metrics:")
    print(f"  Cumulative Return:    {metrics['cumulative_return']:>10.2%}")
    print(f"  Annualized Return:    {metrics['annualized_return']:>10.2%}")
    
    print(f"\n⚠️  Risk Metrics:")
    print(f"  Volatility (Ann.):    {metrics['volatility']:>10.2%}")
    print(f"  Sharpe Ratio:         {metrics['sharpe_ratio']:>10.4f}")
    print(f"  Max Drawdown:         {metrics['max_drawdown']:>10.2%}")
    
    print(f"\n✅ Coverage Metrics:")
    print(f"  Coverage Rate:        {metrics['coverage_rate']:>10.2%}")
    print(f"  Violation Rate:       {metrics['violation_rate']:>10.2%}")
    
    print(f"\n💼 Trading Metrics:")
    print(f"  Avg Turnover:         {metrics['turnover']:>10.2%}")
    
    print(f"\n⏱️  Computational Metrics:")
    print(f"  Avg Solve Time:       {metrics['avg_solve_time']:>10.4f}s")
    print(f"  Total Solve Time:     {metrics['total_solve_time']:>10.2f}s")
    
    print(f"{'='*60}\n")

def compare_portfolios(portfolios_metrics: Dict[str, Dict]) -> pd.DataFrame:
    """Compare multiple portfolios"""
    df = pd.DataFrame(portfolios_metrics).T
    
    column_order = [
        'cumulative_return', 'annualized_return', 'volatility', 
        'sharpe_ratio', 'max_drawdown', 'coverage_rate', 'violation_rate',
        'turnover', 'avg_solve_time', 'total_solve_time'
    ]
    
    column_order = [col for col in column_order if col in df.columns]
    df = df[column_order]
    
    return df