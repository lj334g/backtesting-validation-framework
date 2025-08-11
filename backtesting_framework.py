"""
Trading Strategy Backtesting Framework
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ta
import yfinance as yf
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


@dataclass
class BacktestConfig:
    """Configuration parameters for backtesting"""
    initial_capital: float = 1_000_000
    risk_per_trade: float = 0.02
    commission_rate: float = 0.005
    bid_ask_spread: float = 0.0002
    market_impact_factor: float = 0.0001
    risk_free_rate: float = 0.02
    trading_days_per_year: int = 252


class DataLoader:
    """Handles market data loading and preprocessing"""
    
    @staticmethod
    def load_market_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Load historical market data"""
        data = yf.download(symbol, start=start_date, end=end_date)
        
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
        
        return data


class Strategy(ABC):
    """Abstract base class for trading strategies"""
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals for given data"""
        pass


class SMAStrategy(Strategy):
    """Simple Moving Average crossover strategy"""
    
    def __init__(self, short_window: int = 50, long_window: int = 200):
        self.short_window = short_window
        self.long_window = long_window
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate SMA crossover signals"""
        result = data.copy()
        sma_short = ta.trend.sma_indicator(result['Close'], window=self.short_window)
        sma_long = ta.trend.sma_indicator(result['Close'], window=self.long_window)
        result['signal'] = np.where(sma_short > sma_long, 1, -1)
        return result


class TransactionCostModel:
    """Calculates realistic transaction costs"""
    
    def __init__(self, config: BacktestConfig):
        self.commission_rate = config.commission_rate
        self.bid_ask_spread = config.bid_ask_spread
        self.market_impact_factor = config.market_impact_factor
    
    def calculate_transaction_cost(self, price: float, quantity: float, avg_volume: float) -> Tuple[float, Dict]:
        """Calculate total transaction cost and breakdown"""
        trade_value = abs(quantity * price)
        
        commission = max(abs(quantity) * self.commission_rate, 1.0)
        spread_cost = trade_value * (self.bid_ask_spread / 2)
        
        volume_participation = abs(quantity) / max(avg_volume, 1000)
        market_impact = trade_value * self.market_impact_factor * min(volume_participation, 0.1)
        
        slippage = trade_value * np.random.normal(0, 0.0001)
        
        total_cost = commission + spread_cost + market_impact + abs(slippage)
        
        return total_cost, {
            'commission': commission,
            'spread_cost': spread_cost,
            'market_impact': market_impact,
            'slippage': abs(slippage)
        }


class PortfolioManager:
    """Manages portfolio state and trade execution"""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.cash = config.initial_capital
        self.positions = {}
        self.trades = []
        self.transaction_costs = []
        self.cost_model = TransactionCostModel(config)
    
    def execute_trade(self, symbol: str, quantity: float, price: float, avg_volume: float) -> None:
        """Execute a trade with transaction costs"""
        total_cost, cost_breakdown = self.cost_model.calculate_transaction_cost(
            price, quantity, avg_volume
        )
        
        self.cash -= (quantity * price + total_cost)
        self.positions[symbol] = self.positions.get(symbol, 0) + quantity
        
        trade_record = {
            'symbol': symbol,
            'quantity': quantity,
            'price': price,
            'total_cost': total_cost,
            'cost_breakdown': cost_breakdown
        }
        self.trades.append(trade_record)
        self.transaction_costs.append(total_cost)
    
    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """Calculate current portfolio value"""
        positions_value = sum(
            self.positions.get(symbol, 0) * price 
            for symbol, price in current_prices.items()
        )
        return self.cash + positions_value
    
    def reset(self) -> None:
        """Reset portfolio to initial state"""
        self.cash = self.config.initial_capital
        self.positions = {}
        self.trades = []
        self.transaction_costs = []


class BacktestEngine:
    """Main backtesting engine"""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.portfolio_manager = PortfolioManager(config)
    
    def run_backtest(self, data: pd.DataFrame, symbol: str) -> Dict:
        """Execute backtest on given data"""
        self.portfolio_manager.reset()
        
        avg_volume = data.get('Volume', pd.Series([1_000_000] * len(data))).mean()
        
        for index, row in data.iterrows():
            self._process_signal(row, symbol, avg_volume)
        
        return self._generate_results(data, symbol)
    
    def _process_signal(self, row: pd.Series, symbol: str, avg_volume: float) -> None:
        """Process trading signal for current row"""
        close_price = row['Close']
        signal = row['signal']
        
        if signal == 1:
            available_cash = self.portfolio_manager.cash
            position_size = available_cash * self.config.risk_per_trade / close_price
            self.portfolio_manager.execute_trade(symbol, position_size, close_price, avg_volume)
        elif signal == -1:
            current_position = self.portfolio_manager.positions.get(symbol, 0)
            if current_position > 0:
                self.portfolio_manager.execute_trade(symbol, -current_position, close_price, avg_volume)
    
    def _generate_results(self, data: pd.DataFrame, symbol: str) -> Dict:
        """Generate backtest results"""
        portfolio_values = []
        for _, row in data.iterrows():
            portfolio_value = self.portfolio_manager.get_portfolio_value({symbol: row['Close']})
            portfolio_values.append(portfolio_value)
        
        return {
            'portfolio_values': portfolio_values,
            'trades': self.portfolio_manager.trades,
            'transaction_costs': self.portfolio_manager.transaction_costs,
            'final_value': portfolio_values[-1],
            'initial_value': portfolio_values[0]
        }


class PerformanceCalculator:
    """Calculates performance metrics"""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
    
    def calculate_metrics(self, results: Dict, data: pd.DataFrame) -> Dict:
        """Calculate comprehensive performance metrics"""
        portfolio_values = results['portfolio_values']
        initial_value = results['initial_value']
        final_value = results['final_value']
        
        total_return = (final_value - initial_value) / initial_value
        annualized_return = (1 + total_return) ** (self.config.trading_days_per_year / len(data)) - 1
        
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        volatility = np.std(returns) * np.sqrt(self.config.trading_days_per_year)
        sharpe_ratio = (annualized_return - self.config.risk_free_rate) / volatility if volatility > 0 else 0
        
        trade_metrics = self._calculate_trade_metrics(results['trades'])
        cost_analysis = self._calculate_cost_analysis(results['transaction_costs'], initial_value)
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            **trade_metrics,
            **cost_analysis
        }
    
    def _calculate_trade_metrics(self, trades: List[Dict]) -> Dict:
        """Calculate trade-specific metrics"""
        if len(trades) < 2:
            return {'total_pnl': 0, 'average_trade_return': 0, 'win_ratio': 0}
        
        trade_prices = np.array([trade['price'] for trade in trades])
        trade_quantities = np.array([trade['quantity'] for trade in trades])
        
        trade_returns = np.diff(trade_prices) / trade_prices[:-1]
        trade_pnl = trade_returns * trade_quantities[:-1]
        
        return {
            'total_pnl': np.sum(trade_pnl),
            'average_trade_return': np.mean(trade_returns),
            'win_ratio': np.sum(trade_pnl > 0) / len(trade_pnl) if len(trade_pnl) > 0 else 0
        }
    
    def _calculate_cost_analysis(self, transaction_costs: List[float], initial_value: float) -> Dict:
        """Calculate transaction cost analysis"""
        total_costs = sum(transaction_costs)
        cost_drag = total_costs / initial_value
        
        return {
            'total_transaction_costs': total_costs,
            'cost_drag': cost_drag
        }


class MarketRegimeAnalyzer:
    """Analyzes strategy performance across market regimes"""
    
    def __init__(self, strategy: Strategy, backtest_engine: BacktestEngine):
        self.strategy = strategy
        self.backtest_engine = backtest_engine
    
    def analyze_regimes(self, data: pd.DataFrame, symbol: str) -> Dict:
        """Analyze strategy performance across different market regimes"""
        regimes = self._identify_regimes(data)
        regime_performance = {}
        
        for regime_name in regimes['regime'].unique():
            if regime_name == 'Unknown':
                continue
                
            regime_mask = regimes['regime'] == regime_name
            regime_indices = regimes[regime_mask].index
            
            if len(regime_indices) < 50:
                continue
            
            regime_data = data.loc[regime_indices].copy()
            regime_data = self.strategy.generate_signals(regime_data)
            
            results = self.backtest_engine.run_backtest(regime_data, symbol)
            regime_return = (results['final_value'] - results['initial_value']) / results['initial_value']
            
            regime_performance[regime_name] = {
                'return': regime_return,
                'trades': len(results['trades']),
                'days': len(regime_data)
            }
        
        return regime_performance
    
    def _identify_regimes(self, data: pd.DataFrame) -> pd.DataFrame:
        """Identify market regimes based on volatility and returns"""
        data_copy = data.copy()
        data_copy['rolling_vol'] = data_copy['Close'].pct_change().rolling(60).std() * np.sqrt(252)
        data_copy['rolling_return'] = data_copy['Close'].pct_change(60)
        
        regimes = []
        for _, row in data_copy.iterrows():
            vol = row['rolling_vol']
            ret = row['rolling_return']
            
            if pd.isna(vol) or pd.isna(ret):
                regime = 'Unknown'
            elif ret > 0.1:
                regime = 'Bull Market'
            elif ret < -0.1:
                regime = 'Bear Market'
            elif vol > 0.3:
                regime = 'High Volatility'
            else:
                regime = 'Normal Market'
            
            regimes.append(regime)
        
        return pd.DataFrame({'regime': regimes}, index=data_copy.index)


class ValidationFramework:
    """Implements validation techniques to prevent overfitting"""
    
    def __init__(self, strategy: Strategy, backtest_engine: BacktestEngine, performance_calculator: PerformanceCalculator):
        self.strategy = strategy
        self.backtest_engine = backtest_engine
        self.performance_calculator = performance_calculator
    
    def walk_forward_analysis(self, data: pd.DataFrame, symbol: str, 
                            train_periods: int = 120, test_periods: int = 30, step_size: int = 15) -> List[Dict]:
        """Perform walk-forward analysis"""
        results = []
        
        for start in range(0, len(data) - train_periods - test_periods, step_size):
            test_start = start + train_periods
            test_end = test_start + test_periods
            
            if test_end > len(data):
                break
            
            test_data = data.iloc[test_start:test_end].copy()
            test_data = self.strategy.generate_signals(test_data)
            
            backtest_results = self.backtest_engine.run_backtest(test_data, symbol)
            metrics = self.performance_calculator.calculate_metrics(backtest_results, test_data)
            
            results.append({
                'test_start': test_data.index[0],
                'test_end': test_data.index[-1],
                'return': metrics['total_return'],
                'trades': len(backtest_results['trades'])
            })
        
        return results
    
    def train_test_split(self, data: pd.DataFrame, symbol: str, split_ratio: float = 0.7) -> Dict:
        """Perform train/test split validation"""
        split_point = int(len(data) * split_ratio)
        
        train_data = data.iloc[:split_point].copy()
        test_data = data.iloc[split_point:].copy()
        test_data = self.strategy.generate_signals(test_data)
        
        backtest_results = self.backtest_engine.run_backtest(test_data, symbol)
        metrics = self.performance_calculator.calculate_metrics(backtest_results, test_data)
        
        return {
            'train_period': f"{train_data.index[0]} to {train_data.index[-1]}",
            'test_period': f"{test_data.index[0]} to {test_data.index[-1]}",
            'test_return': metrics['total_return'],
            'test_trades': len(backtest_results['trades']),
            'signal_distribution': test_data['signal'].value_counts().to_dict()
        }


class ResultsVisualizer:
    """Handles visualization of backtest results"""
    
    @staticmethod
    def plot_portfolio_performance(data: pd.DataFrame, portfolio_values: List[float]) -> None:
        """Plot portfolio value over time with signals"""
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        ax1.plot(data.index, portfolio_values, label='Portfolio Value')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Portfolio Value')
        ax1.set_title('Portfolio Value Over Time')
        
        ax2 = ax1.twinx()
        ax2.plot(data.index, data['signal'], 'r-', label='Buy/Sell Signal')
        ax2.set_ylabel('Signal')
        ax2.grid(None)
        
        fig.tight_layout()
        plt.show()


class BacktestOrchestrator:
    """Main orchestrator that coordinates all components"""
    
    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()
        self.strategy = SMAStrategy()
        self.backtest_engine = BacktestEngine(self.config)
        self.performance_calculator = PerformanceCalculator(self.config)
        self.validation_framework = ValidationFramework(
            self.strategy, self.backtest_engine, self.performance_calculator
        )
        self.regime_analyzer = MarketRegimeAnalyzer(self.strategy, self.backtest_engine)
    
    def run_full_analysis(self, symbol: str, start_date: str, end_date: str) -> None:
        """Run complete backtesting analysis"""
        data = DataLoader.load_market_data(symbol, start_date, end_date)
        data = self.strategy.generate_signals(data)
        
        results = self.backtest_engine.run_backtest(data, symbol)
        metrics = self.performance_calculator.calculate_metrics(results, data)
        
        self._print_performance_metrics(metrics)
        self._run_validation_analysis(data, symbol)
        self._run_regime_analysis(data, symbol)
        
        ResultsVisualizer.plot_portfolio_performance(data, results['portfolio_values'])
    
    def _print_performance_metrics(self, metrics: Dict) -> None:
        """Print performance metrics"""
        print('--- Performance Metrics ---')
        print(f'Total Returns: {metrics["total_return"]:.2%}')
        print(f'Annualized Returns: {metrics["annualized_return"]:.2%}')
        print(f'Volatility: {metrics["volatility"]:.2%}')
        print(f'Sharpe Ratio: {metrics["sharpe_ratio"]:.2f}')
        print(f'Total P&L: {metrics["total_pnl"]:.2f}')
        print(f'Average Trade Return: {metrics["average_trade_return"]:.2%}')
        print(f'Win Ratio: {metrics["win_ratio"]:.2%}')
        
        print('\n--- Transaction Cost Analysis ---')
        print(f'Total Transaction Costs: ${metrics["total_transaction_costs"]:.2f}')
        print(f'Cost Drag on Returns: {metrics["cost_drag"]:.4%}')
        print(f'Return without costs: {(metrics["total_return"] + metrics["cost_drag"]):.2%}')
        print(f'Return with costs: {metrics["total_return"]:.2%}')
    
    def _run_validation_analysis(self, data: pd.DataFrame, symbol: str) -> None:
        """Run validation analysis"""
        print('\n--- Validation Analysis ---')
        
        wf_results = self.validation_framework.walk_forward_analysis(data, symbol)
        if wf_results:
            wf_returns = [r['return'] for r in wf_results]
            print(f'Out-of-sample periods tested: {len(wf_results)}')
            print(f'Average out-of-sample return: {np.mean(wf_returns):.2%}')
            print(f'Std dev of returns: {np.std(wf_returns):.2%}')
            print(f'Positive periods: {(np.array(wf_returns) > 0).mean():.1%}')
        else:
            print('No walk-forward results - data period may be too short')
        
        split_results = self.validation_framework.train_test_split(data, symbol)
        print(f'\nTrain period: {split_results["train_period"]}')
        print(f'Test period: {split_results["test_period"]}')
        print(f'Out-of-sample return: {split_results["test_return"]:.2%}')
        print(f'Test period signal distribution: {split_results["signal_distribution"]}')
        
        if split_results["test_return"] == 0 and split_results["test_trades"] == 0:
            print('Strategy demonstrates overfitting - no trades in test period')
    
    def _run_regime_analysis(self, data: pd.DataFrame, symbol: str) -> None:
        """Run market regime analysis"""
        print('\n--- Market Regime Analysis ---')
        try:
            regime_performance = self.regime_analyzer.analyze_regimes(data, symbol)
            
            if regime_performance:
                for regime, perf in regime_performance.items():
                    print(f'{regime}: {perf["return"]:.2%} return, {perf["trades"]} trades, {perf["days"]} days')
                
                regime_returns = [perf['return'] for perf in regime_performance.values()]
                print(f'\nAverage regime return: {np.mean(regime_returns):.2%}')
                print(f'Regime return volatility: {np.std(regime_returns):.2%}')
                print(f'Positive regimes: {(np.array(regime_returns) > 0).mean():.1%}')
            else:
                print('No regime analysis available - insufficient data for regime identification')
        except Exception as e:
            print(f'Regime analysis failed: {str(e)}')


def main():
    print("Trading Strategy Backtesting Framework")
    
    symbol = input("Ticker: ")
    start_date = input("Start Date (YYYY-MM-DD): ")
    end_date = input("End Date (YYYY-MM-DD): ")
    
    orchestrator = BacktestOrchestrator()
    orchestrator.run_full_analysis(symbol, start_date, end_date)


if __name__ == '__main__':
    main()
