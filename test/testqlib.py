import pytest
import torch
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from alphagen_qlib.stock_data import StockData, FeatureType
from alphagen_qlib.calculator import QLibStockDataCalculator
from alphagen_qlib.strategy import TopKSwapNStrategy, TradeDecisionWO
from alphagen_generic.operators import Mean, Ref

@pytest.fixture
def mock_qlib_init():
    with patch('alphagen_qlib.stock_data.initialize_qlib') as mock_init:
        yield mock_init

@pytest.fixture
def sample_stock_data():
    dates = pd.date_range(start="2020-01-01", end="2020-01-10")
    stock_ids = pd.Index([f"stock_{i}" for i in range(5)])
    data = torch.randn((len(dates), len(FeatureType), len(stock_ids)))
    return StockData(
        instrument="csi500",
        start_time="2020-01-01",
        end_time="2020-01-10",
        # preloaded_data=(data, dates, stock_ids)
    )

class TestStockData:
    def test_data_slicing(self, sample_stock_data):
        sliced = sample_stock_data[1:5]
        assert sliced.n_days == 4
        assert sliced.n_stocks == sample_stock_data.n_stocks
        assert sliced.n_features == sample_stock_data.n_features

    def test_date_indexing(self, sample_stock_data):
        idx = sample_stock_data.find_date_index("2020-01-05")
        assert idx == 4
        with pytest.raises(ValueError):
            sample_stock_data.find_date_index("2019-12-31")

    def test_invalid_data_handling(self, mock_qlib_init):
        with pytest.raises(Exception):
            StockData(
                instrument="invalid",
                start_time="2020-01-01",
                end_time="2020-01-10",
                qlib_path="./invalid_path"
            )

class TestQLibCalculator:
    def test_normalization(self, sample_stock_data):
        calculator = QLibStockDataCalculator(sample_stock_data)
        expr = Mean(Ref(FeatureType.CLOSE, -1), window=5)
        result = calculator.evaluate_alpha(expr)
        
        # Check normalization
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()
        assert (result.std(dim=0) > 0).all()  # All stocks should have non-zero std

    def test_empty_expression(self, sample_stock_data):
        calculator = QLibStockDataCalculator(sample_stock_data)
        with pytest.raises(AttributeError):
            calculator.evaluate_alpha(None)

class TestTopKStrategy:
    @pytest.mark.parametrize("K,n_swap", [(5,2), (10,3), (20,5)])
    def test_order_generation(self, K, n_swap, sample_stock_data):
        strategy = TopKSwapNStrategy(K=K, n_swap=n_swap)
        
        # Mock trade calendar and exchange
        strategy.trade_calendar = MagicMock()
        strategy.trade_exchange = MagicMock()
        strategy.trade_position = MagicMock()
        
        # Mock signal data
        signal_data = {f"stock_{i}": float(i) for i in range(sample_stock_data.n_stocks)}
        strategy.signal = MagicMock()
        strategy.signal.get_signal.return_value = pd.Series(signal_data)
        
        decision = strategy.generate_trade_decision()
        assert isinstance(decision, TradeDecisionWO)
        assert len(decision.order_list) >= n_swap

    def test_edge_cases(self, sample_stock_data):
        # Test K=0 case
        strategy = TopKSwapNStrategy(K=0, n_swap=0)
        decision = strategy.generate_trade_decision()
        assert len(decision.order_list) == 0

        # Test n_swap > K case
        strategy = TopKSwapNStrategy(K=5, n_swap=10)
        decision = strategy.generate_trade_decision()
        assert len(decision.order_list) <= 5

if __name__ == "__main__":
    pytest.main(["-v", "--cov=alphagen_qlib", "--cov-report=html"])