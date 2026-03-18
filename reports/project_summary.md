# Project Summary

- Project objective: estimate calibrated NFL win probabilities from bookmaker odds and evaluate whether those probabilities support a positive expected value selection rule.
- Modeling approach: de vig implied probabilities, sharp book consensus priors, beta and BBQ calibration models, and EV ranking by bookmaker and outcome.
- Key metrics: Brier score, log loss, expected calibration error, hit rate, cumulative profit, and return on stake.
- Limitations: the included backtest allows stacked bets on the same game and does not model limits, slippage, or market impact; live snapshot collection requires an API key.
- Possible extensions: grouped cross validation by week, bankroll aware staking, correlated bet controls, and calibration drift monitoring over time.
