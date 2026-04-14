#!/usr/bin/env python3
"""
Backtest ML Filter on Historical Trades

Runs the trained ML filter on historical trade data to measure
what the filter would have done.

Usage:
    python tools/backtest_ml_filter.py
    python tools/backtest_ml_filter.py --threshold 0.6
    python tools/backtest_ml_filter.py --model models/trade_quality_model.joblib
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.ml.trade_quality_filter import TradeQualityFilter


@dataclass
class TradeResult:
    """Result of a single trade."""
    trade_id: str
    symbol: str
    strategy: str
    entry_price: float
    exit_price: float
    pnl: float
    won: bool
    ml_score: float
    ml_approved: bool
    features: Dict[str, float]


def load_trades_with_features(path: str) -> List[Dict]:
    """Load completed trades that have features."""
    entries = {}
    exits = {}

    with open(path, 'r') as f:
        for line in f:
            try:
                event = json.loads(line.strip())
                trade_id = event.get('trade_id')

                if event.get('event') == 'entry':
                    entries[trade_id] = event
                elif event.get('event') == 'exit':
                    exits[trade_id] = event
            except json.JSONDecodeError:
                continue

    # Merge entries with exits, keeping only trades with features
    trades = []
    for trade_id, entry in entries.items():
        if trade_id not in exits:
            continue

        features = entry.get('features', {})
        # Skip trades without ML features
        if not any(k.startswith('feat_') or k in ['atr_percentile', 'hour_of_day'] for k in features.keys()):
            # Try to extract features from available data
            features = extract_features_from_entry(entry)
            if not features:
                continue

        exit_event = exits[trade_id]
        pnl = exit_event.get('outcome', {}).get('pnl_dollars', 0)

        trades.append({
            'trade_id': trade_id,
            'symbol': entry['symbol'],
            'strategy': features.get('strategy', 'unknown'),
            'entry_price': entry['price'],
            'exit_price': exit_event['price'],
            'pnl': pnl,
            'won': pnl > 0,
            'features': features,
        })

    return trades


def extract_features_from_entry(entry: Dict) -> Dict[str, float]:
    """Extract or infer features from entry data."""
    features = entry.get('features', {})

    # Map raw features to feat_ prefixed names
    result = {}

    # Direct mappings
    mappings = {
        'atr_percentile': 'feat_atr_percentile',
        'drawdown_portfolio_pct': 'feat_drawdown_portfolio_pct',
        'drawdown_symbol_pct': 'feat_drawdown_symbol_pct',
        'position_size_pct': 'feat_position_size_pct',
        'hour_of_day': 'feat_hour_of_day',
        'day_of_week': 'feat_day_of_week',
        'minutes_since_open': 'feat_minutes_since_open',
        'bars_in_regime': 'feat_bars_in_regime',
        'hours_since_last_trade': 'feat_hours_since_last_trade',
        'signal_strength': 'feat_signal_strength',
    }

    for src, dst in mappings.items():
        if src in features:
            result[dst] = features[src]

    # Also check for already prefixed features
    for k, v in features.items():
        if k.startswith('feat_'):
            result[k] = v

    return result if len(result) >= 3 else {}  # Need at least 3 features


def run_backtest(
    trades: List[Dict],
    model_path: str,
    threshold: float
) -> Tuple[List[TradeResult], Dict[str, Any]]:
    """Run ML filter on historical trades."""
    # Load filter
    ml_filter = TradeQualityFilter(
        model_path=model_path,
        min_confidence=threshold,
        enabled=True,
    )

    if not ml_filter.enabled:
        print(f"Error: Could not load model from {model_path}")
        return [], {}

    results = []
    for trade in trades:
        features = trade['features']

        # Evaluate with filter
        approved, score = ml_filter.evaluate(features, symbol=trade['symbol'])

        results.append(TradeResult(
            trade_id=trade['trade_id'],
            symbol=trade['symbol'],
            strategy=trade['strategy'],
            entry_price=trade['entry_price'],
            exit_price=trade['exit_price'],
            pnl=trade['pnl'],
            won=trade['won'],
            ml_score=score,
            ml_approved=approved,
            features=features,
        ))

    # Calculate statistics
    stats = calculate_stats(results, threshold)
    return results, stats


def calculate_stats(results: List[TradeResult], threshold: float) -> Dict[str, Any]:
    """Calculate backtest statistics."""
    if not results:
        return {'error': 'No results'}

    total = len(results)
    total_wins = sum(1 for r in results if r.won)
    total_pnl = sum(r.pnl for r in results)

    # Split by filter decision
    approved = [r for r in results if r.ml_approved]
    rejected = [r for r in results if not r.ml_approved]

    # Approved stats
    approved_wins = sum(1 for r in approved if r.won)
    approved_pnl = sum(r.pnl for r in approved)

    # Rejected stats
    rejected_wins = sum(1 for r in rejected if r.won)
    rejected_losses = sum(1 for r in rejected if not r.won)
    rejected_pnl = sum(r.pnl for r in rejected)

    # Rates
    baseline_win_rate = total_wins / total if total > 0 else 0
    approved_win_rate = approved_wins / len(approved) if approved else 0
    rejected_win_rate = rejected_wins / len(rejected) if rejected else 0

    # Score distribution
    scores = [r.ml_score for r in results]
    winner_scores = [r.ml_score for r in results if r.won]
    loser_scores = [r.ml_score for r in results if not r.won]

    return {
        'threshold': threshold,
        'total_trades': total,

        'baseline': {
            'trades': total,
            'wins': total_wins,
            'losses': total - total_wins,
            'win_rate': baseline_win_rate,
            'total_pnl': total_pnl,
            'avg_pnl': total_pnl / total if total > 0 else 0,
        },

        'with_filter': {
            'trades_taken': len(approved),
            'trades_blocked': len(rejected),
            'block_rate': len(rejected) / total if total > 0 else 0,
        },

        'approved': {
            'trades': len(approved),
            'wins': approved_wins,
            'losses': len(approved) - approved_wins,
            'win_rate': approved_win_rate,
            'total_pnl': approved_pnl,
            'avg_pnl': approved_pnl / len(approved) if approved else 0,
        },

        'rejected': {
            'trades': len(rejected),
            'wins': rejected_wins,
            'losses': rejected_losses,
            'win_rate': rejected_win_rate,
            'total_pnl': rejected_pnl,
            'avg_pnl': rejected_pnl / len(rejected) if rejected else 0,
        },

        'effectiveness': {
            'win_rate_improvement': approved_win_rate - baseline_win_rate,
            'pnl_improvement': approved_pnl - total_pnl,  # Would have saved this
            'correct_rejections': rejected_losses,
            'incorrect_rejections': rejected_wins,
            'rejection_accuracy': rejected_losses / len(rejected) if rejected else 0,
        },

        'score_analysis': {
            'avg_score': sum(scores) / len(scores) if scores else 0,
            'avg_winner_score': sum(winner_scores) / len(winner_scores) if winner_scores else 0,
            'avg_loser_score': sum(loser_scores) / len(loser_scores) if loser_scores else 0,
            'score_separation': (sum(winner_scores) / len(winner_scores) if winner_scores else 0) -
                               (sum(loser_scores) / len(loser_scores) if loser_scores else 0),
        },
    }


def print_report(stats: Dict[str, Any], show_thresholds: bool = False) -> None:
    """Print formatted backtest report."""
    print("\n" + "=" * 65)
    print("ML FILTER BACKTEST RESULTS")
    print("=" * 65)

    if 'error' in stats:
        print(f"\nError: {stats['error']}")
        return

    print(f"\nThreshold: {stats['threshold']}")
    print(f"Total trades analyzed: {stats['total_trades']}")

    # Baseline
    print("\n" + "-" * 45)
    print("BASELINE (No Filter)")
    print("-" * 45)
    b = stats['baseline']
    print(f"  Trades:   {b['trades']}")
    print(f"  Wins:     {b['wins']} ({b['win_rate']:.1%})")
    print(f"  Losses:   {b['losses']}")
    print(f"  Total P&L: ${b['total_pnl']:,.2f}")
    print(f"  Avg P&L:   ${b['avg_pnl']:,.2f}")

    # With Filter
    print("\n" + "-" * 45)
    print("WITH ML FILTER")
    print("-" * 45)
    wf = stats['with_filter']
    print(f"  Trades Taken:   {wf['trades_taken']}")
    print(f"  Trades Blocked: {wf['trades_blocked']} ({wf['block_rate']:.1%})")

    # Approved
    print("\n  APPROVED (would trade):")
    a = stats['approved']
    print(f"    Trades:   {a['trades']}")
    print(f"    Wins:     {a['wins']} ({a['win_rate']:.1%})")
    print(f"    Total P&L: ${a['total_pnl']:,.2f}")
    print(f"    Avg P&L:   ${a['avg_pnl']:,.2f}")

    # Rejected
    print("\n  REJECTED (would block):")
    r = stats['rejected']
    print(f"    Trades:   {r['trades']}")
    print(f"    Wins:     {r['wins']} ({r['win_rate']:.1%})")
    print(f"    Losses:   {r['losses']}")
    print(f"    Total P&L: ${r['total_pnl']:,.2f}")
    print(f"    Avg P&L:   ${r['avg_pnl']:,.2f}")

    # Effectiveness
    print("\n" + "-" * 45)
    print("FILTER EFFECTIVENESS")
    print("-" * 45)
    e = stats['effectiveness']
    print(f"  Win Rate Improvement:    {e['win_rate_improvement']:+.1%}")
    print(f"  Correct Rejections:      {e['correct_rejections']} (blocked losers)")
    print(f"  Incorrect Rejections:    {e['incorrect_rejections']} (blocked winners)")
    print(f"  Rejection Accuracy:      {e['rejection_accuracy']:.1%}")

    # P&L analysis
    if r['total_pnl'] < 0:
        print(f"  P&L Saved by Blocking:   ${-r['total_pnl']:,.2f}")
    else:
        print(f"  P&L Lost by Blocking:    ${r['total_pnl']:,.2f}")

    # Score Analysis
    print("\n" + "-" * 45)
    print("SCORE ANALYSIS")
    print("-" * 45)
    s = stats['score_analysis']
    print(f"  Avg Score (all):     {s['avg_score']:.3f}")
    print(f"  Avg Score (winners): {s['avg_winner_score']:.3f}")
    print(f"  Avg Score (losers):  {s['avg_loser_score']:.3f}")
    print(f"  Score Separation:    {s['score_separation']:.3f}")

    # Recommendation
    print("\n" + "=" * 65)
    print("RECOMMENDATION")
    print("=" * 65)

    improvement = e['win_rate_improvement']
    if improvement > 0.10:
        print(f"  ✓✓ Strong improvement ({improvement:+.1%}) - Enable the filter")
    elif improvement > 0.05:
        print(f"  ✓  Good improvement ({improvement:+.1%}) - Consider enabling")
    elif improvement > 0:
        print(f"  ~  Modest improvement ({improvement:+.1%}) - May help")
    else:
        print(f"  ✗  No improvement ({improvement:+.1%}) - Keep disabled")

    if s['score_separation'] > 0.1:
        print(f"  ✓  Good score separation ({s['score_separation']:.3f}) - Model discriminates well")
    else:
        print(f"  ✗  Poor score separation ({s['score_separation']:.3f}) - Model struggles to discriminate")

    print()


def sweep_thresholds(
    trades: List[Dict],
    model_path: str,
    thresholds: List[float]
) -> None:
    """Test multiple thresholds and show comparison."""
    print("\n" + "=" * 75)
    print("THRESHOLD SWEEP")
    print("=" * 75)
    print(f"\n{'Threshold':<12} {'Trades':<10} {'Win Rate':<12} {'Improvement':<14} {'P&L':<15}")
    print("-" * 75)

    # First show baseline
    _, baseline_stats = run_backtest(trades, model_path, 0.0)
    if 'error' in baseline_stats:
        print("Error running backtest")
        return

    baseline_wr = baseline_stats['baseline']['win_rate']
    baseline_pnl = baseline_stats['baseline']['total_pnl']
    print(f"{'Baseline':<12} {baseline_stats['baseline']['trades']:<10} {baseline_wr:<12.1%} {'---':<14} ${baseline_pnl:<14,.2f}")
    print("-" * 75)

    for threshold in thresholds:
        _, stats = run_backtest(trades, model_path, threshold)
        if 'error' in stats:
            continue

        trades_taken = stats['approved']['trades']
        win_rate = stats['approved']['win_rate']
        improvement = stats['effectiveness']['win_rate_improvement']
        pnl = stats['approved']['total_pnl']

        marker = "  ✓" if improvement > 0.05 else "   "
        print(f"{threshold:<12.2f} {trades_taken:<10} {win_rate:<12.1%} {improvement:<+14.1%} ${pnl:<14,.2f}{marker}")

    print()


def main():
    parser = argparse.ArgumentParser(description="Backtest ML Filter on historical trades")
    parser.add_argument('--trades', default='logs/meta_trades_live.jsonl',
                        help='Path to trade data file')
    parser.add_argument('--model', default='models/trade_quality_model.joblib',
                        help='Path to ML model')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Confidence threshold (default: 0.5)')
    parser.add_argument('--sweep', action='store_true',
                        help='Test multiple thresholds')
    parser.add_argument('--json', action='store_true',
                        help='Output as JSON')

    args = parser.parse_args()

    # Check files exist
    trades_path = Path(args.trades)
    model_path = Path(args.model)

    if not trades_path.exists():
        print(f"Trades file not found: {trades_path}")
        return

    if not model_path.exists():
        print(f"Model not found: {model_path}")
        print("Train a model first: python tools/train_trade_model.py")
        return

    # Load trades
    print(f"Loading trades from {trades_path}...")
    trades = load_trades_with_features(str(trades_path))
    print(f"  Found {len(trades)} trades with features")

    if not trades:
        print("No trades with features found. Cannot backtest.")
        return

    # Threshold sweep
    if args.sweep:
        sweep_thresholds(
            trades,
            str(model_path),
            [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        )
        return

    # Single threshold backtest
    print(f"\nRunning backtest with threshold={args.threshold}...")
    results, stats = run_backtest(trades, str(model_path), args.threshold)

    if args.json:
        print(json.dumps(stats, indent=2, default=str))
    else:
        print_report(stats)


if __name__ == '__main__':
    main()
