"""
Tests for Risk Metrics Module

Coverage:
- StandardRiskModel
  - compute_metrics
  - compute_metric
  - Individual metric calculations
  - Beta/alpha calculation
- RiskQuantifier (legacy)
  - All calculation methods
  - risk_profile
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.risk.metrics import RiskQuantifier, StandardRiskModel


class TestStandardRiskModelInit:
    """Tests for StandardRiskModel initialization."""

    def test_default_init(self):
        """Test default initialization."""
        model = StandardRiskModel()

        assert model.risk_free_rate == 0.0  # Default from RiskModelBase
        assert model.periods_per_year == 252  # Default for daily

    def test_custom_init(self):
        """Test custom initialization."""
        model = StandardRiskModel(risk_free_rate=0.05, periods_per_year=365)

        assert model.risk_free_rate == 0.05
        assert model.periods_per_year == 365


class TestComputeMetrics:
    """Tests for compute_metrics method."""

    @pytest.fixture
    def model(self):
        """Create a StandardRiskModel."""
        return StandardRiskModel(risk_free_rate=0.04)

    @pytest.fixture
    def sample_returns(self):
        """Create sample returns series."""
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.02, 252))  # 1 year of daily returns
        return returns

    def test_compute_metrics_returns_dict(self, model, sample_returns):
        """Test that compute_metrics returns a dictionary."""
        metrics = model.compute_metrics(sample_returns)

        assert isinstance(metrics, dict)
        assert "sharpe_ratio" in metrics
        assert "sortino_ratio" in metrics
        assert "max_drawdown" in metrics
        assert "var_95" in metrics

    def test_compute_metrics_all_keys(self, model, sample_returns):
        """Test that all expected keys are present."""
        metrics = model.compute_metrics(sample_returns)

        expected_keys = [
            "total_return",
            "annualized_return",
            "volatility",
            "volatility_daily",
            "sharpe_ratio",
            "sortino_ratio",
            "calmar_ratio",
            "max_drawdown",
            "current_drawdown",
            "var_95",
            "var_99",
            "cvar_95",
            "win_rate",
            "profit_factor",
            "n_periods",
        ]

        for key in expected_keys:
            assert key in metrics, f"Missing key: {key}"

    def test_compute_metrics_with_benchmark(self, model, sample_returns):
        """Test compute_metrics with benchmark returns."""
        np.random.seed(43)
        benchmark = pd.Series(np.random.normal(0.0005, 0.015, 252))

        metrics = model.compute_metrics(sample_returns, benchmark_returns=benchmark)

        assert "beta" in metrics
        assert "alpha" in metrics

    def test_compute_metrics_empty_returns(self, model):
        """Test compute_metrics with insufficient data."""
        short_returns = pd.Series([0.01])  # Only 1 data point

        metrics = model.compute_metrics(short_returns)

        # Should return empty metrics
        assert metrics["n_periods"] == 0
        assert metrics["sharpe_ratio"] == 0.0


class TestComputeMetric:
    """Tests for compute_metric method."""

    @pytest.fixture
    def model(self):
        return StandardRiskModel()

    @pytest.fixture
    def sample_returns(self):
        np.random.seed(42)
        return pd.Series(np.random.normal(0.001, 0.02, 100))

    def test_compute_single_metric(self, model, sample_returns):
        """Test computing a single metric."""
        sharpe = model.compute_metric(sample_returns, "sharpe_ratio")

        assert isinstance(sharpe, (int, float))

    def test_compute_metric_var(self, model, sample_returns):
        """Test computing VaR metric."""
        var95 = model.compute_metric(sample_returns, "var_95")

        assert isinstance(var95, (int, float))
        assert var95 < 0  # VaR should be negative (loss)

    def test_compute_metric_cvar(self, model, sample_returns):
        """Test computing CVaR metric."""
        cvar95 = model.compute_metric(sample_returns, "cvar_95")

        assert isinstance(cvar95, (int, float))

    def test_compute_metric_unknown(self, model, sample_returns):
        """Test computing unknown metric raises error."""
        with pytest.raises(ValueError, match="Unknown metric"):
            model.compute_metric(sample_returns, "unknown_metric")


class TestIndividualMetrics:
    """Tests for individual metric calculation methods."""

    @pytest.fixture
    def model(self):
        return StandardRiskModel(risk_free_rate=0.04)

    def test_compute_total_return(self, model):
        """Test total return calculation."""
        returns = pd.Series([0.01, 0.02, -0.01, 0.015])

        total = model.compute_total_return(returns)

        # (1+0.01)*(1+0.02)*(1-0.01)*(1+0.015) - 1
        expected = (1.01 * 1.02 * 0.99 * 1.015) - 1
        assert abs(total - expected) < 0.0001

    def test_compute_annualized_return(self, model):
        """Test annualized return calculation."""
        # 252 days of 0.1% daily returns
        returns = pd.Series([0.001] * 252)

        annual = model.compute_annualized_return(returns)

        # Should be positive and annualized
        assert annual > 0

    def test_compute_volatility(self, model):
        """Test volatility calculation."""
        returns = pd.Series([0.01, -0.01, 0.02, -0.02, 0.01])

        vol = model.compute_volatility(returns)

        assert vol > 0

    def test_compute_volatility_annualized(self, model):
        """Test annualized volatility is higher than daily."""
        returns = pd.Series([0.01, -0.01, 0.02, -0.02, 0.01])

        vol_annual = model.compute_volatility(returns, annualized=True)
        vol_daily = model.compute_volatility(returns, annualized=False)

        assert vol_annual > vol_daily

    def test_compute_sharpe_ratio_positive(self, model):
        """Test Sharpe ratio with positive returns."""
        # Consistently positive returns
        returns = pd.Series([0.01] * 100)

        sharpe = model.compute_sharpe_ratio(returns)

        assert sharpe > 0

    def test_compute_sharpe_ratio_negative(self, model):
        """Test Sharpe ratio with negative returns."""
        # Consistently negative returns
        returns = pd.Series([-0.01] * 100)

        sharpe = model.compute_sharpe_ratio(returns)

        assert sharpe < 0

    def test_compute_sortino_ratio(self, model):
        """Test Sortino ratio calculation."""
        # Mix of positive and negative returns
        returns = pd.Series([0.02, -0.01, 0.015, -0.005, 0.01])

        sortino = model.compute_sortino_ratio(returns)

        # Should be defined
        assert not np.isnan(sortino)

    def test_compute_max_drawdown(self, model):
        """Test max drawdown calculation."""
        # Create returns that produce a known drawdown
        returns = pd.Series([0.10, -0.20, 0.05, -0.10, 0.15])

        mdd = model.compute_max_drawdown(returns)

        assert mdd < 0  # Drawdown is negative
        assert mdd >= -1  # Can't lose more than 100%

    def test_compute_current_drawdown(self, model):
        """Test current drawdown calculation."""
        # Returns ending below peak
        returns = pd.Series([0.10, 0.05, -0.02, -0.03])

        current_dd = model.compute_current_drawdown(returns)

        assert current_dd <= 0

    def test_compute_var(self, model):
        """Test Value at Risk calculation."""
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0, 0.02, 100))

        var95 = model.compute_var(returns, 0.95)
        var99 = model.compute_var(returns, 0.99)

        # 99% VaR should be more extreme (more negative) than 95% VaR
        assert var99 <= var95

    def test_compute_cvar(self, model):
        """Test Conditional VaR (Expected Shortfall)."""
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0, 0.02, 100))

        var95 = model.compute_var(returns, 0.95)
        cvar95 = model.compute_cvar(returns, 0.95)

        # CVaR should be more extreme than VaR
        assert cvar95 <= var95

    def test_compute_win_rate(self, model):
        """Test win rate calculation."""
        # 3 wins, 2 losses
        returns = pd.Series([0.01, 0.02, -0.01, 0.015, -0.005])

        win_rate = model.compute_win_rate(returns)

        assert win_rate == 0.6  # 3/5

    def test_compute_profit_factor(self, model):
        """Test profit factor calculation."""
        # Gains: 0.01 + 0.02 + 0.015 = 0.045
        # Losses: 0.01 + 0.005 = 0.015
        returns = pd.Series([0.01, 0.02, -0.01, 0.015, -0.005])

        pf = model.compute_profit_factor(returns)

        assert pf == pytest.approx(3.0, rel=0.01)  # 0.045 / 0.015 = 3

    def test_compute_calmar_ratio(self, model):
        """Test Calmar ratio calculation."""
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.02, 252))

        calmar = model.compute_calmar_ratio(returns)

        # Should be defined
        assert not np.isnan(calmar)


class TestBetaAlpha:
    """Tests for beta and alpha calculation."""

    @pytest.fixture
    def model(self):
        return StandardRiskModel(risk_free_rate=0.04)

    def test_beta_alpha_calculation(self, model):
        """Test beta and alpha calculation."""
        np.random.seed(42)
        # Create correlated returns
        benchmark = pd.Series(np.random.normal(0.001, 0.015, 100))
        strategy = benchmark * 1.2 + pd.Series(np.random.normal(0.0002, 0.005, 100))

        beta, alpha = model._compute_beta_alpha(strategy, benchmark)

        # Beta should be around 1.2 (our multiplier)
        assert 0.8 < beta < 1.6

    def test_beta_alpha_insufficient_data(self, model):
        """Test beta/alpha with insufficient data."""
        short_strat = pd.Series([0.01, 0.02])
        short_bench = pd.Series([0.005, 0.01])

        beta, alpha = model._compute_beta_alpha(short_strat, short_bench)

        assert beta == 0.0
        assert alpha == 0.0


class TestEmptyMetrics:
    """Tests for empty metrics dictionary."""

    def test_empty_metrics(self):
        """Test empty metrics dictionary."""
        model = StandardRiskModel()
        metrics = model._empty_metrics()

        assert metrics["total_return"] == 0.0
        assert metrics["sharpe_ratio"] == 0.0
        assert metrics["n_periods"] == 0


class TestRiskQuantifier:
    """Tests for legacy RiskQuantifier class."""

    @pytest.fixture
    def quantifier(self):
        """Create a RiskQuantifier."""
        return RiskQuantifier()

    @pytest.fixture
    def returns_df(self):
        """Create sample returns DataFrame."""
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 100)
        return pd.DataFrame({"return": returns})

    def test_set_returns(self, quantifier, returns_df):
        """Test setting returns data."""
        quantifier.set_returns(returns_df, "return")

        assert quantifier.returns is not None
        assert len(quantifier.returns) == 100

    def test_set_returns_invalid_column(self, quantifier, returns_df):
        """Test setting returns with invalid column."""
        with pytest.raises(ValueError, match="not found"):
            quantifier.set_returns(returns_df, "invalid_column")

    def test_set_returns_not_dataframe(self, quantifier):
        """Test setting returns with non-DataFrame."""
        with pytest.raises(ValueError, match="must be a Pandas DataFrame"):
            quantifier.set_returns([0.01, 0.02], "return")

    def test_set_risk_free_rate(self, quantifier):
        """Test setting risk-free rate."""
        quantifier.set_risk_free_rate(0.05)

        assert quantifier.risk_free_rate == 0.05

    def test_calculate_standard_deviation(self, quantifier, returns_df):
        """Test standard deviation calculation."""
        quantifier.set_returns(returns_df, "return")

        std = quantifier.calculate_standard_deviation()

        assert std > 0

    def test_calculate_standard_deviation_no_data(self, quantifier):
        """Test standard deviation without data raises error."""
        with pytest.raises(ValueError, match="Returns data not set"):
            quantifier.calculate_standard_deviation()

    def test_calculate_sharpe_ratio(self, quantifier, returns_df):
        """Test Sharpe ratio calculation."""
        quantifier.set_returns(returns_df, "return")

        sharpe = quantifier.calculate_sharpe_ratio()

        assert isinstance(sharpe, float)

    def test_calculate_sortino_ratio(self, quantifier, returns_df):
        """Test Sortino ratio calculation."""
        quantifier.set_returns(returns_df, "return")

        sortino = quantifier.calculate_sortino_ratio()

        # May be nan if no downside returns, but should compute
        assert isinstance(sortino, float) or np.isnan(sortino)

    def test_calculate_max_drawdown(self, quantifier, returns_df):
        """Test max drawdown calculation."""
        quantifier.set_returns(returns_df, "return")

        mdd = quantifier.calculate_max_drawdown()

        assert mdd <= 0

    def test_calculate_kelly_criterion(self, quantifier, returns_df):
        """Test Kelly criterion calculation."""
        quantifier.set_returns(returns_df, "return")

        kelly = quantifier.calculate_kelly_criterion()

        assert isinstance(kelly, float)

    def test_calculate_var(self, quantifier, returns_df):
        """Test VaR calculation."""
        quantifier.set_returns(returns_df, "return")

        var = quantifier.calculate_var(confidence_level=0.05)

        assert var < 0  # Should be negative (loss)

    def test_calculate_beta(self, quantifier, returns_df):
        """Test beta calculation."""
        quantifier.set_returns(returns_df, "return")
        market_df = returns_df.copy()

        beta = quantifier.calculate_beta(market_df)

        # With same returns, beta should be 1
        assert abs(beta - 1.0) < 0.1

    def test_calculate_alpha(self, quantifier, returns_df):
        """Test alpha calculation."""
        quantifier.set_returns(returns_df, "return")
        market_df = returns_df.copy()

        alpha = quantifier.calculate_alpha(market_df)

        # With same returns, alpha should be near 0
        assert abs(alpha) < 0.01

    def test_calculate_treynor_ratio(self, quantifier, returns_df):
        """Test Treynor ratio calculation."""
        quantifier.set_returns(returns_df, "return")
        market_df = returns_df.copy()

        treynor = quantifier.calculate_treynor_ratio(market_df)

        assert isinstance(treynor, float)

    def test_risk_profile(self, quantifier):
        """Test risk profile calculation."""
        np.random.seed(42)
        df = pd.DataFrame({"Strategy_Return": np.random.normal(0.001, 0.02, 100)})

        profile = quantifier.risk_profile(df, "TestStrategy")

        assert "Sharpe Ratio" in profile
        assert "Max Drawdown" in profile
        assert "VaR" in profile
        assert profile["Strategy"] == "TestStrategy"

    def test_risk_profile_missing_column(self, quantifier):
        """Test risk profile with missing column."""
        df = pd.DataFrame({"Other_Return": [0.01, 0.02]})

        with pytest.raises(ValueError, match="Strategy_Return"):
            quantifier.risk_profile(df, "TestStrategy")
