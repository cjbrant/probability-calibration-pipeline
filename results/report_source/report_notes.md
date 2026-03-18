# Figure 1   Home vs Away: Actual Win Rate vs Bookmaker Prediction (2020–2025)

## What this figure shows

This figure compares the actual win rates of home and away teams with the true probabilities estimated from de-vigged betting odds over the five-year period from 2020 to 2025.

To avoid duplicated games from multiple sportsbooks, the data were grouped by game date, home team, and away team, and only one record per game was kept.

## What we calculated

- The actual home win rate based on real game outcomes
- The actual away win rate, calculated as one minus the home win rate
- The average true home probability and average true away probability, estimated from betting odds after removing the bookmaker margin (vig)

## Key Results
- The home team actual win rate over five years is 53.9%
- The away team actual win rate is 46.1%
- The average true probability for the home team is 54.9%
- The average true probability for the away team is 45.1%

## What this chart shows

Home teams do win slightly more often than away teams in real games. More importantly, the true probabilities estimated from the betting market after removing the vig are very close to the actual historical win rates.

This indicates that, after removing the bookmaker’s margin, the betting market provides a fairly accurate estimate of real win probabilities. Therefore, it is reasonable to use these de-vigged probabilities in our EV-based betting strategy.


# Figure 2  Are Betting Odds Accurate? (2020–2025)What this figure shows

## What this figure shows

This figure evaluates the calibration accuracy of sportsbook odds using NFL data from 2020 to 2025.

All games are grouped into probability bins based on the home team’s implied probability.

Within each bin, we compare:
- The actual home win rate
- The average implied probability derived from odds

The black dashed diagonal line represents perfect calibration, where predicted probability equals the actual win rate.

## How to interpret the plot

- Points close to the dashed line indicate that the odds are well calibrated
- Points below the dashed line indicate that the sportsbook overestimates the win probability
- Points above the dashed line indicate that the sportsbook underestimates the win probability


## Key Findings from the 5-Year Data (2020–2025)

1. Highest accuracy occurs in the mid-probability range (50%–70%)
   
In this range, actual win rates closely match implied probabilities. This indicates that sportsbooks price balanced and moderately favored games with the highest precision, where market efficiency is strongest.

2. Low-probability home teams are systematically overestimated
   
When the home team has a low implied probability, the actual win rate is typically lower than predicted. This suggests that sportsbooks tend to overestimate underdogs, likely influenced by bettor preference for high-payout outcomes.

3. Heavy favorites are slightly underestimated
   
For very high implied probabilities, actual win rates are often slightly higher than predicted. This suggests that sportsbooks modestly undervalue extreme favorites, possibly due to risk management considerations.

## Core Conclusion

Sportsbook odds are generally well calibrated across most probability ranges, especially in the mid-range where market efficiency is highest.
However, systematic pricing biases appear at probability extremes: underdogs are overestimated, while heavy favorites are underestimated.


# Figure 3   Which Bookmaker Gives Best EV? (2020–2025)
 
## What this figure shows
This figure compares average expected value (EV) across different sportsbooks using NFL data from 2020–2025.

For each bookmaker, we:
- Calculate the mean home EV and away EV using our de-vigged true probabilities
- Take their average to get a single overall EV per bookmaker
- Sort bookmakers by this average EV and plot a horizontal bar chart
- Draw a vertical line at EV = 0; bars to the left of this line are negative EV for the bettor
Bars are colored green if average EV is positive and red if it is negative (in our data, all are red).

## How to interpret the plot
- Each bar shows how good or bad it is (on average) to bet with that bookmaker, assuming our true-probability model is correct.
- Values closer to 0 indicate better prices for the bettor (smaller house edge).
- More negative values mean the bookmaker is “tighter” or more expensive, offering worse long-run returns to a naïve bettor.

## Key Findings from 2020–2025

1. All books have negative average EV
- No sportsbook offers positive EV on average across all games.
- This is expected: the house retains an edge in the long run.
  
2. Some books are “less bad” than others
- A few bookmakers have average EV closest to zero, meaning they offer relatively better odds (smaller house edge).
  Others show much more negative EV, implying consistently worse pricing for bettors.

3. Bookmaker choice matters even if the game is the same
- For the same game, betting at a sharper/cheaper book can significantly reduce the long-run loss rate.
- This highlights the value of line shopping: choosing the bookmaker that gives the best price instead of always using just one.
	
## Core Conclusion
On average, every sportsbook still has a negative EV for the bettor, but the size of that edge varies a lot by bookmaker.

From an EV perspective, a rational bettor should:  Avoid books with very negative average EV and prefer those whose average EV is closest to zero, since they consistently offer better prices for the same underlying games.


# Figure 4   +EV Betting Strategy: Cumulative Profit (2020–2025)

## What this figure shows

This figure simulates the performance of a simple value betting strategy applied to NFL games from 2020 to 2025.

The strategy is defined as follows:
- A bet is placed whenever a positive expected value (EV) is observed at any individual sportsbook
- Each qualifying opportunity receives a fixed stake of $100
- If the home team wins, profit is calculated as
(odds - 1)*100
- If the home team loses, the loss is –$100
- Profits are accumulated sequentially to produce the cumulative profit curve

Under this definition, multiple bets may occur on the same game if several sportsbooks simultaneously offer positive EV on that matchup.
In total, the strategy placed 2,913 +EV bets over the five-year period.

## Key Results
- Total number of +EV bets: 2,913
- Win rate: 44.6%
- Total profit: $21,216
Although the win rate is below 50%, the strategy is profitable because many wins occur at favorable odds, which is consistent with the interpretation of positive expected value.

How to interpret the curve
- The x-axis represents the number of betting opportunities executed
- The y-axis shows the cumulative profit in dollars
- The dashed horizontal line at zero represents the break-even level
Despite short-term volatility, the long-term trend remains positive, indicating that the +EV strategy is profitable over a large sample.

## Core Conclusion and Limitation
This backtest demonstrates that consistently betting on positive expected value opportunities can generate long-term profit, even with a sub-50% win rate. 

However, an important limitation of this simulation is that it operates at the bookmaker level rather than the game level. Because a separate bet is placed whenever any sportsbook offers positive EV, the same game may be wagered multiple times at different sportsbooks. This makes the strategy more aggressive and higher in risk exposure than a realistic approach where a bettor would typically select only one sportsbook per game.

As a result, the cumulative profit shown here likely represents an upper-bound performance for a value-based strategy, rather than a directly implementable real-world strategy without further risk control.
