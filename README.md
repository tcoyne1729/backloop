# backloop

Simple multi-asset backtesting framework.

## Install

```
uv sync
```

## Concepts

**Data** — a `pd.DataFrame` with a `DatetimeIndex`. Each column is an asset identifier; values are close prices.

**Strategy** — subclass `Strategy` and implement `generate_signals()`. It receives the price DataFrame and must return a DataFrame of the same shape where each value is the target portfolio weight for that asset on that date. Weights should be non-negative and sum to ≤ 1 per row.

**Backtester** — takes the data, a strategy class, and optional parameters. On each bar it rebalances to the target weights. Fractional weights are floored to integer share counts.

## Usage

```python
import pandas as pd
from main import Strategy, Backtester

class MyStrategy(Strategy):
    def generate_signals(self) -> pd.DataFrame:
        # return DataFrame of target weights, same index and columns as self.data
        ...

bt = Backtester(
    data,                        # pd.DataFrame, columns = asset ids
    MyStrategy,                  # class, not instance
    initial_cash=10_000,
    commission_fn=lambda p, u: p * u * 0.001,  # optional, defaults to zero
    position_sizing=0.9,         # fraction of equity to deploy, default 1.0
)

result = bt.run()   # returns data with an Equity column appended
bt.stats()          # returns dict of performance metrics
bt.stats(risk_free_rate=0.04)  # Sharpe/Sortino risk-free rate, default 4%
```

## Metrics returned by `stats()`

| Key | Description |
|-----|-------------|
| CAGR | Annualised compounded return |
| Total Return | Simple return over the full period |
| Annualised Volatility | Std dev of daily returns * sqrt(252) |
| Sharpe Ratio | (CAGR - rfr) / ann. vol |
| Sortino Ratio | (CAGR - rfr) / downside vol |
| Calmar Ratio | CAGR / \|max drawdown\| |
| Max Drawdown | Worst peak-to-trough as a fraction |
| Max Drawdown Duration (days) | Longest consecutive period underwater |
| Average Drawdown | Mean of all underwater values |
| Current Drawdown | Drawdown at the final bar |
| Num Trades | Total rebalancing events |
| Avg Holding Period (days) | Mean FIFO round-trip duration |
| Avg Profit per Trade | Mean P&L across FIFO round trips |
| Avg Win / Avg Loss | Mean P&L of winning / losing round trips |

## Trades

`bt.trades` is a list of `(date, asset, price, delta_units)` tuples — one entry per rebalancing event.
