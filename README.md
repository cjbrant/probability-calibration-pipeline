# Probability Calibration Pipeline

Finding positive expected value bets by calibrating sportsbook odds against sharp book consensus.

## What this is

This project was started with a friend who was curious about exploiting price differences between sportsbooks. The core question: can you systematically find bets where the implied probability from a sportsbook is wrong enough to bet against?

In sports betting, "sharp" books (like Pinnacle) set accurate lines, and the rest of the market adjusts to follow... but with a lag. This project looks for that lag: cases where a recreational book's price implies a different probability than the sharp consensus, creating a positive expected value opportunity.

## What it does

The pipeline:

1. Pulls odds from multiple sportsbooks via The Odds API
2. Strips the vig (bookmaker margin) to get implied probabilities
3. Builds a consensus "true" probability from the sharp books
4. Calibrates these probabilities using Beta calibration and BBQ (Bayesian Binning into Quantiles) (two methods for correcting systematic bias in probability estimates)
5. Scans for bets where a book's price implies a lower win probability than our calibrated estimate
6. Backtests the strategy on 5 years of NFL moneyline data

## Key findings

- The market is very efficient: home teams won 53.9% of games vs. the market's 54.9% prediction (only 1% gap)
- But there's bias at the extremes: underdogs are overpriced, heavy favorites are slightly underpriced
- Calibration tightens probability estimates, especially in the tails where the bias concentrates
- Backtest: 2,913 +EV bets, 44.6% win rate, $21,216 cumulative profit, 7.28% ROI on flat $100 stakes
- Important caveat: the backtest assumes stacking (multiple bets per game across books), which inflates both returns and risk

## The takeaway

Probability calibration matters whenever you're making decisions based on estimated probabilities. Even in a highly efficient market like NFL betting, calibrating the extremes of the probability distribution reveals exploitable patterns. The methods here (Beta calibration, BBQ) apply anywhere you need well-calibrated probabilities, not just sports.

## How to run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# Fit calibration models on training data
python -m evbets fit-calibrator --data data/training/demo_training.parquet --method bbq --model models/bbq.joblib

# Scan a snapshot for +EV bets
python -m evbets scan --in data/snapshots/americanfootball_nfl_20251126_031409.json --market h2h --odds-format american --sharp-books pinnacle --sharp-books betonlineag --target-books pinnacle --target-books betonlineag --no-log

# Run the full backtest
python -m evbets backtest
```

## Tech stack

Python, pandas, numpy, scipy, scikit-learn, joblib. CLI built with standard argparse. Calibration uses Beta calibration and BBQ implementations. Data from The Odds API.

## Structure

- `src/evbets/` — core library (odds conversion, consensus, calibration, CLI)
- `data/` — snapshots, training data, results templates
- `results/` — figures and report source
- `reports/` — calibrated probabilities and +EV scan outputs
- `models/` — fitted calibration model artifacts
