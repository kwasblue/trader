"""
Backtest Comparison Engine - Compare strategies, categories, and hybrid sizing.

This module provides tools for comparing multiple backtesting approaches:
1. Compare specific strategies head-to-head
2. Compare strategy categories (trend-following vs mean-reversion)
3. Compare hybrid vs standard position sizing
4. Generate comprehensive reports

Usage:
    from core.backtest.backtest_comparison import BacktestComparison

    comparison = BacktestComparison(data)

    # Compare specific strategies
    result = comparison.compare_strategies(["sma", "ema", "rsi", "macd"])

    # Compare categories
    result = comparison.compare_categories(["trend_following", "mean_reversion"])

    # Compare hybrid sizing
    result = comparison.compare_hybrid(["sma", "macd", "rsi"])

    # Generate report
    report = comparison.generate_report(result)
    print(report.to_markdown())
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import pandas as pd
import numpy as np
import logging

from core.backtest.unified_backtest_runner import (
    UnifiedBacktestRunner,
    BacktestConfig,
    BacktestResult,
    BacktestMetrics,
    STRATEGY_CATEGORIES,
    get_strategy_category,
    get_strategies_by_category,
    list_available_strategies,
)

logger = logging.getLogger(__name__)


@dataclass
class StrategyComparisonResult:
    """Result of comparing multiple strategies."""

    strategies: List[str]
    results: Dict[str, BacktestResult]
    rankings: Dict[str, int]  # strategy -> rank
    best_strategy: str
    best_metric_value: float
    metric_used: str
    data_period: str
    num_bars: int


@dataclass
class CategoryComparisonResult:
    """Result of comparing strategy categories."""

    categories: List[str]
    category_results: Dict[str, Dict[str, Any]]  # category -> avg metrics
    strategy_results: Dict[str, BacktestResult]  # individual strategy results
    best_category: str
    rankings: Dict[str, int]  # category -> rank


@dataclass
class HybridComparisonResult:
    """Result of comparing hybrid vs standard sizing."""

    strategies: List[str]
    standard_results: Dict[str, BacktestResult]
    hybrid_results: Dict[str, BacktestResult]
    improvements: Dict[str, float]  # strategy -> return delta
    best_improvement: str
    trade_distribution: Dict[str, Dict[str, int]]  # strategy -> {with_trend, against_trend}


@dataclass
class ComparisonReport:
    """Full comparison report with multiple sections."""

    title: str
    symbol: str
    date_generated: str
    data_period: str
    num_bars: int

    strategy_comparison: Optional[StrategyComparisonResult] = None
    category_comparison: Optional[CategoryComparisonResult] = None
    hybrid_comparison: Optional[HybridComparisonResult] = None

    insights: List[str] = field(default_factory=list)

    def to_markdown(self) -> str:
        """Generate markdown report."""
        lines = [
            f"# {self.title}",
            "",
            f"**Symbol:** {self.symbol}",
            f"**Generated:** {self.date_generated}",
            f"**Data Period:** {self.data_period}",
            f"**Bars Analyzed:** {self.num_bars}",
            "",
        ]

        # Strategy comparison section
        if self.strategy_comparison:
            lines.extend(self._format_strategy_comparison())

        # Category comparison section
        if self.category_comparison:
            lines.extend(self._format_category_comparison())

        # Hybrid comparison section
        if self.hybrid_comparison:
            lines.extend(self._format_hybrid_comparison())

        # Insights
        if self.insights:
            lines.extend([
                "## Key Insights",
                "",
            ])
            for insight in self.insights:
                lines.append(f"- {insight}")
            lines.append("")

        return "\n".join(lines)

    def _format_strategy_comparison(self) -> List[str]:
        """Format strategy comparison section."""
        sc = self.strategy_comparison
        lines = [
            "## Strategy Comparison",
            "",
            "| Strategy | Return | Sharpe | Max DD | Win Rate | Trades | Rank |",
            "|----------|--------|--------|--------|----------|--------|------|",
        ]

        # Sort by ranking
        sorted_strategies = sorted(sc.strategies, key=lambda s: sc.rankings.get(s, 999))

        for strat in sorted_strategies:
            result = sc.results.get(strat)
            if result:
                m = result.metrics
                rank = sc.rankings.get(strat, "-")
                lines.append(
                    f"| {strat} | {m.total_return:+.1%} | {m.sharpe_ratio:.2f} | "
                    f"{m.max_drawdown:.1%} | {m.win_rate:.1%} | {m.num_trades} | #{rank} |"
                )

        lines.extend(["", f"**Best Strategy:** {sc.best_strategy}", ""])
        return lines

    def _format_category_comparison(self) -> List[str]:
        """Format category comparison section."""
        cc = self.category_comparison
        lines = [
            "## Category Comparison",
            "",
            "| Category | Avg Return | Avg Sharpe | Best Strategy |",
            "|----------|------------|------------|---------------|",
        ]

        sorted_categories = sorted(cc.categories, key=lambda c: cc.rankings.get(c, 999))

        for cat in sorted_categories:
            cat_data = cc.category_results.get(cat, {})
            avg_return = cat_data.get("avg_return", 0)
            avg_sharpe = cat_data.get("avg_sharpe", 0)
            best_strat = cat_data.get("best_strategy", "-")
            rank = cc.rankings.get(cat, "-")

            lines.append(
                f"| {cat.replace('_', ' ').title()} | {avg_return:+.1%} | "
                f"{avg_sharpe:.2f} | {best_strat} |"
            )

        lines.extend(["", f"**Best Category:** {cc.best_category.replace('_', ' ').title()}", ""])
        return lines

    def _format_hybrid_comparison(self) -> List[str]:
        """Format hybrid comparison section."""
        hc = self.hybrid_comparison
        lines = [
            "## Hybrid vs Standard Sizing",
            "",
            "| Strategy | Standard | Hybrid | Delta | Improved? |",
            "|----------|----------|--------|-------|-----------|",
        ]

        for strat in hc.strategies:
            std_result = hc.standard_results.get(strat)
            hyb_result = hc.hybrid_results.get(strat)

            if std_result and hyb_result:
                std_return = std_result.metrics.total_return
                hyb_return = hyb_result.metrics.total_return
                delta = hc.improvements.get(strat, 0)
                improved = "YES" if delta > 0 else "NO"
                category = get_strategy_category(strat)

                lines.append(
                    f"| {strat} | {std_return:+.1%} | {hyb_return:+.1%} | "
                    f"{delta:+.1%} | {improved} ({category}) |"
                )

        lines.append("")

        # Trade distribution
        lines.extend([
            "### Trade Distribution (Hybrid)",
            "",
        ])

        total_with = 0
        total_against = 0
        wins_with = 0
        wins_against = 0

        for strat in hc.strategies:
            hyb_result = hc.hybrid_results.get(strat)
            if hyb_result:
                m = hyb_result.metrics
                total_with += m.trades_with_trend
                total_against += m.trades_against_trend

                # Approximate wins
                if m.trades_with_trend > 0:
                    wins_with += int(m.trades_with_trend * m.win_rate_with_trend)
                if m.trades_against_trend > 0:
                    wins_against += int(m.trades_against_trend * m.win_rate_against_trend)

        total = total_with + total_against
        if total > 0:
            pct_with = total_with / total * 100
            pct_against = total_against / total * 100
            wr_with = wins_with / total_with * 100 if total_with > 0 else 0
            wr_against = wins_against / total_against * 100 if total_against > 0 else 0

            lines.extend([
                f"- Trades with trend: {total_with} ({pct_with:.0f}%) - {wr_with:.1f}% win rate",
                f"- Trades against trend: {total_against} ({pct_against:.0f}%) - {wr_against:.1f}% win rate",
                "",
            ])

        return lines

    def to_json(self) -> str:
        """Generate JSON report."""
        data = {
            "title": self.title,
            "symbol": self.symbol,
            "date_generated": self.date_generated,
            "data_period": self.data_period,
            "num_bars": self.num_bars,
            "insights": self.insights,
        }

        if self.strategy_comparison:
            sc = self.strategy_comparison
            data["strategy_comparison"] = {
                "strategies": sc.strategies,
                "rankings": sc.rankings,
                "best_strategy": sc.best_strategy,
                "metric_used": sc.metric_used,
                "results": {
                    s: r.metrics.to_dict()
                    for s, r in sc.results.items()
                },
            }

        if self.category_comparison:
            cc = self.category_comparison
            data["category_comparison"] = {
                "categories": cc.categories,
                "category_results": cc.category_results,
                "rankings": cc.rankings,
                "best_category": cc.best_category,
            }

        if self.hybrid_comparison:
            hc = self.hybrid_comparison
            data["hybrid_comparison"] = {
                "strategies": hc.strategies,
                "improvements": hc.improvements,
                "best_improvement": hc.best_improvement,
                "standard_results": {
                    s: r.metrics.to_dict()
                    for s, r in hc.standard_results.items()
                },
                "hybrid_results": {
                    s: r.metrics.to_dict()
                    for s, r in hc.hybrid_results.items()
                },
            }

        return json.dumps(data, indent=2)


class BacktestComparison:
    """
    Engine for comparing multiple backtesting approaches.

    Provides methods for:
    - Comparing specific strategies
    - Comparing strategy categories
    - Comparing hybrid vs standard sizing
    - Generating comprehensive reports
    """

    def __init__(
        self,
        data: pd.DataFrame,
        initial_capital: float = 10000.0,
        transaction_cost: float = 0.001,
        symbol: str = "UNKNOWN",
    ):
        """
        Initialize the comparison engine.

        Args:
            data: OHLCV DataFrame
            initial_capital: Starting capital
            transaction_cost: Transaction cost fraction
            symbol: Symbol name for reporting
        """
        self.data = data
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.symbol = symbol

        self.runner = UnifiedBacktestRunner(data)

        # Determine data period
        self.data_period = self._get_data_period()
        self.num_bars = len(data)

        logger.info(f"BacktestComparison initialized: {symbol}, {self.num_bars} bars")

    def _get_data_period(self) -> str:
        """Get human-readable data period."""
        try:
            if "Date" in self.data.columns:
                dates = pd.to_datetime(self.data["Date"])
                return f"{dates.min().date()} to {dates.max().date()}"
            elif isinstance(self.data.index, pd.DatetimeIndex):
                return f"{self.data.index.min().date()} to {self.data.index.max().date()}"
        except Exception:
            pass
        return f"{self.num_bars} bars"

    def compare_strategies(
        self,
        strategies: List[str],
        metric: str = "sharpe_ratio",
        use_hybrid: bool = False,
        **config_kwargs,
    ) -> StrategyComparisonResult:
        """
        Compare multiple strategies.

        Args:
            strategies: List of strategy names to compare
            metric: Metric to use for ranking (sharpe_ratio, total_return, etc.)
            use_hybrid: Whether to use hybrid sizing
            **config_kwargs: Additional BacktestConfig parameters

        Returns:
            StrategyComparisonResult
        """
        logger.info(f"Comparing {len(strategies)} strategies (metric={metric})")

        results: Dict[str, BacktestResult] = {}

        for strategy in strategies:
            try:
                result = self.runner.run_strategy(
                    strategy_name=strategy,
                    use_hybrid=use_hybrid,
                    initial_capital=self.initial_capital,
                    transaction_cost=self.transaction_cost,
                    **config_kwargs,
                )
                results[strategy] = result
                logger.info(f"  {strategy}: return={result.metrics.total_return:.2%}")

            except Exception as e:
                logger.warning(f"  {strategy}: FAILED - {e}")

        # Rank by metric
        rankings = self._rank_by_metric(results, metric)

        # Find best
        best_strategy = min(rankings, key=rankings.get) if rankings else ""
        best_value = 0.0
        if best_strategy and best_strategy in results:
            best_value = getattr(results[best_strategy].metrics, metric, 0.0)

        return StrategyComparisonResult(
            strategies=strategies,
            results=results,
            rankings=rankings,
            best_strategy=best_strategy,
            best_metric_value=best_value,
            metric_used=metric,
            data_period=self.data_period,
            num_bars=self.num_bars,
        )

    def compare_categories(
        self,
        categories: Optional[List[str]] = None,
        metric: str = "sharpe_ratio",
        **config_kwargs,
    ) -> CategoryComparisonResult:
        """
        Compare strategy categories.

        Args:
            categories: Categories to compare (default: all)
            metric: Metric for ranking
            **config_kwargs: Additional config parameters

        Returns:
            CategoryComparisonResult
        """
        if categories is None:
            categories = list(STRATEGY_CATEGORIES.keys())

        logger.info(f"Comparing {len(categories)} categories")

        strategy_results: Dict[str, BacktestResult] = {}
        category_results: Dict[str, Dict[str, Any]] = {}

        for category in categories:
            strategies = get_strategies_by_category(category)
            if not strategies:
                continue

            cat_results = []

            for strategy in strategies:
                try:
                    result = self.runner.run_strategy(
                        strategy_name=strategy,
                        initial_capital=self.initial_capital,
                        transaction_cost=self.transaction_cost,
                        **config_kwargs,
                    )
                    strategy_results[strategy] = result
                    cat_results.append(result)
                    logger.info(f"  {category}/{strategy}: return={result.metrics.total_return:.2%}")

                except Exception as e:
                    logger.warning(f"  {category}/{strategy}: FAILED - {e}")

            if cat_results:
                # Calculate category averages
                avg_return = np.mean([r.metrics.total_return for r in cat_results])
                avg_sharpe = np.mean([r.metrics.sharpe_ratio for r in cat_results])
                avg_drawdown = np.mean([r.metrics.max_drawdown for r in cat_results])

                # Find best in category
                best_in_cat = max(cat_results, key=lambda r: getattr(r.metrics, metric, 0))
                best_strategy = best_in_cat.config.strategy_name

                category_results[category] = {
                    "avg_return": avg_return,
                    "avg_sharpe": avg_sharpe,
                    "avg_drawdown": avg_drawdown,
                    "best_strategy": best_strategy,
                    "num_strategies": len(cat_results),
                }

        # Rank categories
        cat_rankings = {}
        sorted_cats = sorted(
            category_results.keys(),
            key=lambda c: category_results[c].get(f"avg_{metric}" if metric == "sharpe_ratio" else "avg_return", 0),
            reverse=True,
        )
        for rank, cat in enumerate(sorted_cats, 1):
            cat_rankings[cat] = rank

        best_category = sorted_cats[0] if sorted_cats else ""

        return CategoryComparisonResult(
            categories=categories,
            category_results=category_results,
            strategy_results=strategy_results,
            best_category=best_category,
            rankings=cat_rankings,
        )

    def compare_hybrid(
        self,
        strategies: List[str],
        **config_kwargs,
    ) -> HybridComparisonResult:
        """
        Compare hybrid vs standard sizing for strategies.

        Args:
            strategies: Strategies to compare
            **config_kwargs: Additional config parameters

        Returns:
            HybridComparisonResult
        """
        logger.info(f"Comparing hybrid sizing for {len(strategies)} strategies")

        standard_results: Dict[str, BacktestResult] = {}
        hybrid_results: Dict[str, BacktestResult] = {}
        improvements: Dict[str, float] = {}
        trade_distribution: Dict[str, Dict[str, int]] = {}

        for strategy in strategies:
            try:
                std_result, hyb_result = self.runner.compare_hybrid(
                    strategy_name=strategy,
                    initial_capital=self.initial_capital,
                    transaction_cost=self.transaction_cost,
                    **config_kwargs,
                )

                standard_results[strategy] = std_result
                hybrid_results[strategy] = hyb_result

                delta = hyb_result.metrics.total_return - std_result.metrics.total_return
                improvements[strategy] = delta

                trade_distribution[strategy] = {
                    "with_trend": hyb_result.metrics.trades_with_trend,
                    "against_trend": hyb_result.metrics.trades_against_trend,
                }

                logger.info(
                    f"  {strategy}: std={std_result.metrics.total_return:.2%}, "
                    f"hyb={hyb_result.metrics.total_return:.2%}, delta={delta:+.2%}"
                )

            except Exception as e:
                logger.warning(f"  {strategy}: FAILED - {e}")

        # Find best improvement
        best_improvement = max(improvements, key=improvements.get) if improvements else ""

        return HybridComparisonResult(
            strategies=strategies,
            standard_results=standard_results,
            hybrid_results=hybrid_results,
            improvements=improvements,
            best_improvement=best_improvement,
            trade_distribution=trade_distribution,
        )

    def full_comparison(
        self,
        strategies: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        include_hybrid: bool = True,
        metric: str = "sharpe_ratio",
        **config_kwargs,
    ) -> ComparisonReport:
        """
        Run a full comparison with all analysis types.

        Args:
            strategies: Specific strategies to compare (default: all)
            categories: Categories to compare (default: main categories)
            include_hybrid: Whether to include hybrid comparison
            metric: Metric for ranking
            **config_kwargs: Additional config parameters

        Returns:
            ComparisonReport with all sections
        """
        # Default strategies
        if strategies is None:
            available = list_available_strategies()
            # Exclude ML/ensemble by default
            strategies = [s for s in available if s not in ("combined", "logisticregression")]

        # Default categories
        if categories is None:
            categories = ["trend_following", "mean_reversion", "momentum"]

        report = ComparisonReport(
            title=f"Strategy Comparison Report - {self.symbol}",
            symbol=self.symbol,
            date_generated=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            data_period=self.data_period,
            num_bars=self.num_bars,
        )

        # Strategy comparison
        logger.info("Running strategy comparison...")
        report.strategy_comparison = self.compare_strategies(
            strategies=strategies,
            metric=metric,
            **config_kwargs,
        )

        # Category comparison
        logger.info("Running category comparison...")
        report.category_comparison = self.compare_categories(
            categories=categories,
            metric=metric,
            **config_kwargs,
        )

        # Hybrid comparison
        if include_hybrid:
            logger.info("Running hybrid comparison...")
            # Select subset for hybrid comparison
            hybrid_strategies = strategies[:6] if len(strategies) > 6 else strategies
            report.hybrid_comparison = self.compare_hybrid(
                strategies=hybrid_strategies,
                **config_kwargs,
            )

        # Generate insights
        report.insights = self._generate_insights(report)

        return report

    def _rank_by_metric(
        self,
        results: Dict[str, BacktestResult],
        metric: str,
    ) -> Dict[str, int]:
        """Rank results by metric (lower rank = better)."""
        # Sort descending (higher metric = better)
        sorted_strategies = sorted(
            results.keys(),
            key=lambda s: getattr(results[s].metrics, metric, 0),
            reverse=True,
        )

        return {s: rank for rank, s in enumerate(sorted_strategies, 1)}

    def _generate_insights(self, report: ComparisonReport) -> List[str]:
        """Generate insights from comparison results."""
        insights = []

        # Strategy comparison insights
        if report.strategy_comparison:
            sc = report.strategy_comparison
            if sc.best_strategy:
                best_result = sc.results.get(sc.best_strategy)
                if best_result:
                    insights.append(
                        f"Best performing strategy: {sc.best_strategy} "
                        f"(Sharpe: {best_result.metrics.sharpe_ratio:.2f}, "
                        f"Return: {best_result.metrics.total_return:.1%})"
                    )

        # Category comparison insights
        if report.category_comparison:
            cc = report.category_comparison
            if cc.best_category:
                cat_data = cc.category_results.get(cc.best_category, {})
                insights.append(
                    f"Best performing category: {cc.best_category.replace('_', ' ').title()} "
                    f"(Avg Sharpe: {cat_data.get('avg_sharpe', 0):.2f})"
                )

        # Hybrid comparison insights
        if report.hybrid_comparison:
            hc = report.hybrid_comparison

            # Count improvements
            improved = sum(1 for d in hc.improvements.values() if d > 0)
            total = len(hc.improvements)

            if total > 0:
                insights.append(
                    f"Hybrid sizing improved {improved}/{total} strategies"
                )

            # Check if trend-following benefits
            tf_strategies = [s for s in hc.strategies if get_strategy_category(s) == "trend_following"]
            mr_strategies = [s for s in hc.strategies if get_strategy_category(s) == "mean_reversion"]

            tf_improved = sum(1 for s in tf_strategies if hc.improvements.get(s, 0) > 0)
            mr_improved = sum(1 for s in mr_strategies if hc.improvements.get(s, 0) > 0)

            if tf_strategies and mr_strategies:
                if tf_improved > mr_improved:
                    insights.append(
                        "Trend-following strategies benefit more from hybrid sizing"
                    )
                elif mr_improved > tf_improved:
                    insights.append(
                        "Mean-reversion strategies may not benefit from hybrid sizing (they trade against trend by design)"
                    )

        return insights

    def generate_report(
        self,
        strategy_result: Optional[StrategyComparisonResult] = None,
        category_result: Optional[CategoryComparisonResult] = None,
        hybrid_result: Optional[HybridComparisonResult] = None,
    ) -> ComparisonReport:
        """
        Generate a report from existing comparison results.

        Args:
            strategy_result: Strategy comparison result
            category_result: Category comparison result
            hybrid_result: Hybrid comparison result

        Returns:
            ComparisonReport
        """
        report = ComparisonReport(
            title=f"Strategy Comparison Report - {self.symbol}",
            symbol=self.symbol,
            date_generated=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            data_period=self.data_period,
            num_bars=self.num_bars,
            strategy_comparison=strategy_result,
            category_comparison=category_result,
            hybrid_comparison=hybrid_result,
        )

        report.insights = self._generate_insights(report)
        return report


def print_strategy_table(
    results: Dict[str, BacktestResult],
    symbol: str = "SYMBOL",
    title: str = "STRATEGY COMPARISON",
) -> None:
    """Print a formatted strategy comparison table."""
    print()
    print("=" * 70)
    print(f"  {title} - {symbol}")
    print("=" * 70)

    # Header
    print()
    print(f"{'Strategy':<15} {'Return':>10} {'Sharpe':>8} {'Max DD':>8} {'Win Rate':>10} {'Trades':>8}")
    print("-" * 70)

    # Sort by Sharpe ratio
    sorted_strats = sorted(
        results.keys(),
        key=lambda s: results[s].metrics.sharpe_ratio,
        reverse=True,
    )

    for strat in sorted_strats:
        m = results[strat].metrics
        print(
            f"{strat:<15} {m.total_return:>+9.1%} {m.sharpe_ratio:>8.2f} "
            f"{m.max_drawdown:>7.1%} {m.win_rate:>9.1%} {m.num_trades:>8}"
        )

    print("-" * 70)
    print()


def print_hybrid_table(
    hybrid_result: HybridComparisonResult,
    symbol: str = "SYMBOL",
) -> None:
    """Print a formatted hybrid comparison table."""
    print()
    print("=" * 70)
    print(f"  HYBRID vs STANDARD SIZING - {symbol}")
    print("=" * 70)

    print()
    print(f"{'Strategy':<12} {'Standard':>10} {'Hybrid':>10} {'Delta':>10} {'Improved?':>12}")
    print("-" * 70)

    for strat in hybrid_result.strategies:
        std = hybrid_result.standard_results.get(strat)
        hyb = hybrid_result.hybrid_results.get(strat)

        if std and hyb:
            std_ret = std.metrics.total_return
            hyb_ret = hyb.metrics.total_return
            delta = hybrid_result.improvements.get(strat, 0)
            improved = "YES" if delta > 0 else "NO"
            category = get_strategy_category(strat)

            print(
                f"{strat:<12} {std_ret:>+9.1%} {hyb_ret:>+9.1%} {delta:>+9.1%} "
                f"{improved:>5} ({category})"
            )

    print("-" * 70)

    # Trade distribution summary
    total_with = sum(
        r.metrics.trades_with_trend
        for r in hybrid_result.hybrid_results.values()
    )
    total_against = sum(
        r.metrics.trades_against_trend
        for r in hybrid_result.hybrid_results.values()
    )
    total = total_with + total_against

    if total > 0:
        print()
        print("TRADE DISTRIBUTION:")
        print(f"  Trades with trend:     {total_with} ({total_with/total*100:.0f}%)")
        print(f"  Trades against trend:  {total_against} ({total_against/total*100:.0f}%)")

    print("=" * 70)
    print()
