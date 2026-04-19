import numpy as np
import pandas as pd


# ---------------------------
# Strategy Base Class
# ---------------------------
class Strategy:
    def __init__(self, data: pd.DataFrame):
        """Initialise strategy with price data.

        Args:
            data: DataFrame whose columns are asset identifiers and values are
                  close prices.

        """
        self.data = data.copy()

    def generate_signals(self) -> pd.DataFrame:
        """Return target portfolio weights for each bar.

        Must be implemented by subclasses.

        Returns:
            pd.DataFrame with the same index as ``self.data``, columns equal
            to the asset identifiers, and values representing the target weight
            for each asset at each point in time.  Weights should be
            non-negative and sum to at most 1.0 per row.

        """
        raise NotImplementedError("Strategy must implement generate_signals()")


# Example Strategy: Moving Average Crossover
class MovingAverageCross(Strategy):
    """Equal-weight among assets whose short MA is above their long MA."""

    def __init__(self, data: pd.DataFrame, short_window: int = 20, long_window: int = 50):
        """Initialise MovingAverageCross strategy.

        Args:
            data: Multi-asset close-price DataFrame.
            short_window: Lookback period for the fast moving average.
            long_window: Lookback period for the slow moving average.

        """
        super().__init__(data)
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self) -> pd.DataFrame:
        """Generate equal-weight signals for assets with bullish MA crossover.

        Returns:
            DataFrame of target weights; active assets share weight equally.

        """
        raw = pd.DataFrame(index=self.data.index, columns=self.data.columns, dtype=float)
        for asset in self.data.columns:
            short_ma = self.data[asset].rolling(self.short_window).mean()
            long_ma = self.data[asset].rolling(self.long_window).mean()
            raw[asset] = (short_ma > long_ma).astype(float)

        # Normalise: equal weight among active assets, zero when none active
        row_sums = raw.sum(axis=1)
        return raw.div(row_sums.where(row_sums > 0, np.nan), axis=0).fillna(0.0)


# ---------------------------
# Backtester
# ---------------------------
class Backtester:
    """Event-driven backtester that supports multi-asset, weight-based signals."""

    def __init__(
        self,
        data: pd.DataFrame,
        strategy: type[Strategy],
        initial_cash: float = 10_000,
        commission_fn=None,
        position_sizing: float = 1.0,
    ):
        """Initialise the backtester.

        Args:
            data: Multi-asset close-price DataFrame (columns = asset identifiers).
            strategy: Strategy *class* (not instance) to instantiate and run.
            initial_cash: Starting cash balance.
            commission_fn: Callable ``(price, units) -> float`` returning the
                commission cost for a trade.  Defaults to zero commission.
            position_sizing: Fraction of total equity to deploy across all
                positions (0.0–1.0).  The remainder stays in cash.

        """
        self.data = data.copy()
        self.assets = list(data.columns)
        self.strategy = strategy(data)
        self.initial_cash = initial_cash
        self.cash = float(initial_cash)
        self.units: dict[str, float] = {a: 0.0 for a in self.assets}
        self.trades: list[tuple] = []  # (date, asset, price, delta_units)
        self.equity_curve: list[float] = []
        self.commission_fn = commission_fn if commission_fn else self._default_commission
        self.position_sizing = position_sizing

    @staticmethod
    def _default_commission(price: float, units: float) -> float:  # noqa: ARG004
        """Return zero commission (no-cost baseline).

        Args:
            price: Execution price.
            units: Number of units traded.

        Returns:
            float: Commission cost (always 0.0).

        """
        return 0.0

    def run(self) -> pd.DataFrame:
        """Execute the backtest and return the price data with an Equity column.

        Returns:
            Copy of ``self.data`` with an appended ``Equity`` column.

        """
        signals = self.strategy.generate_signals()

        for date, prices_row in self.data.iterrows():
            weights: pd.Series = signals.loc[date]

            # Total equity before rebalance
            equity = self.cash + sum(
                self.units[a] * prices_row[a] for a in self.assets
            )
            deployable = equity * self.position_sizing

            # Rebalance to target weights
            for asset in self.assets:
                price = float(prices_row[asset])
                if price <= 0:
                    continue

                target_units = int(deployable * float(weights[asset]) / price)
                delta = target_units - int(self.units[asset])

                if delta == 0:
                    continue

                commission = self.commission_fn(price, abs(delta))
                self.cash -= delta * price + commission
                self.units[asset] = float(target_units)
                self.trades.append((date, asset, price, delta))

            # Equity after rebalance
            equity = self.cash + sum(
                self.units[a] * prices_row[a] for a in self.assets
            )
            self.equity_curve.append(equity)

        result = self.data.copy()
        result["Equity"] = self.equity_curve
        return result

    _TRADING_DAYS_PER_YEAR: int = 252

    def stats(self, risk_free_rate: float = 0.04) -> dict:
        """Return a full suite of performance metrics for the backtest.

        Args:
            risk_free_rate: Annualised risk-free rate used in Sharpe/Sortino
                calculations.  Defaults to 0.04 (4 %).

        Returns:
            dict containing return, risk-adjusted, drawdown, and trade metrics.

        """
        if not self.equity_curve:
            return {}

        equity = np.array(self.equity_curve)
        initial = float(self.initial_cash)
        final = float(equity[-1])

        # ---- Return metrics ----
        total_return = (final - initial) / initial
        n_days = (self.data.index[-1] - self.data.index[0]).days
        years = n_days / 365.25 if n_days > 0 else 1.0
        cagr = (final / initial) ** (1.0 / years) - 1.0

        daily_returns = np.diff(equity) / equity[:-1]
        ann_vol = float(daily_returns.std() * np.sqrt(self._TRADING_DAYS_PER_YEAR))

        # ---- Risk-adjusted metrics ----
        sharpe = (cagr - risk_free_rate) / ann_vol if ann_vol > 0 else np.nan

        downside = daily_returns[daily_returns < 0]
        sortino_vol = float(downside.std() * np.sqrt(self._TRADING_DAYS_PER_YEAR)) if len(downside) > 1 else np.nan
        sortino = (cagr - risk_free_rate) / sortino_vol if sortino_vol and sortino_vol > 0 else np.nan

        dd_series = self._drawdown_series(equity)
        max_dd = float(dd_series.min()) if len(dd_series) > 0 else 0.0
        calmar = cagr / abs(max_dd) if max_dd != 0 else np.nan

        # ---- Drawdown metrics ----
        max_dd_duration = self._max_drawdown_duration(dd_series)
        below = dd_series[dd_series < 0]
        avg_dd = float(below.mean()) if len(below) > 0 else 0.0
        current_dd = float(dd_series[-1]) if len(dd_series) > 0 else 0.0

        # ---- Trade metrics ----
        round_trips = self._round_trips()
        profits = [rt["profit"] for rt in round_trips]
        holding_days = [rt["holding_days"] for rt in round_trips]
        wins = [p for p in profits if p > 0]
        losses = [p for p in profits if p <= 0]

        return {
            # Return
            "CAGR": cagr,
            "Total Return": total_return,
            "Annualised Volatility": ann_vol,
            # Risk-adjusted
            "Sharpe Ratio": sharpe,
            "Sortino Ratio": sortino,
            "Calmar Ratio": calmar,
            # Drawdown
            "Max Drawdown": max_dd,
            "Max Drawdown Duration (days)": max_dd_duration,
            "Average Drawdown": avg_dd,
            "Current Drawdown": current_dd,
            # Trades
            "Num Trades": len(self.trades),
            "Avg Holding Period (days)": float(np.mean(holding_days)) if holding_days else np.nan,
            "Avg Profit per Trade": float(np.mean(profits)) if profits else np.nan,
            "Win Rate": len(wins) / len(profits) if profits else np.nan,
            "Avg Win": float(np.mean(wins)) if wins else np.nan,
            "Avg Loss": float(np.mean(losses)) if losses else np.nan,
        }

    @staticmethod
    def _drawdown_series(equity: np.ndarray) -> np.ndarray:
        """Return the drawdown fraction at each point in time.

        Args:
            equity: Array of equity values.

        Returns:
            Array of drawdown fractions (≤ 0).

        """
        peak = np.maximum.accumulate(equity)
        return (equity - peak) / peak

    @staticmethod
    def _max_drawdown_duration(dd_series: np.ndarray) -> int:
        """Return the length in bars of the longest drawdown period.

        A drawdown period is any consecutive run of bars where equity is below
        its previous peak.

        Args:
            dd_series: Drawdown fraction array from :meth:`_drawdown_series`.

        Returns:
            int: Maximum number of consecutive bars spent underwater.

        """
        max_dur = cur_dur = 0
        for d in dd_series:
            if d < 0:
                cur_dur += 1
                if cur_dur > max_dur:
                    max_dur = cur_dur
            else:
                cur_dur = 0
        return max_dur

    def _round_trips(self) -> list[dict]:
        """Reconstruct FIFO round-trip trades from the rebalancing event log.

        Returns:
            List of dicts with keys: asset, entry_date, exit_date, profit,
            holding_days.

        """
        lots: dict[str, list[dict]] = {}
        round_trips: list[dict] = []

        for date, asset, price, delta in self.trades:
            if asset not in lots:
                lots[asset] = []

            if delta > 0:  # buy — open a new lot
                lots[asset].append({"date": date, "price": price, "units": delta})

            elif delta < 0:  # sell — close against FIFO lots
                units_to_sell = abs(delta)
                proceeds = units_to_sell * price
                cost = 0.0
                entry_date = None

                while units_to_sell > 0 and lots[asset]:
                    lot = lots[asset][0]
                    if entry_date is None:
                        entry_date = lot["date"]
                    filled = min(lot["units"], units_to_sell)
                    cost += filled * lot["price"]
                    lot["units"] -= filled
                    units_to_sell -= filled
                    if lot["units"] == 0:
                        lots[asset].pop(0)

                if entry_date is not None:
                    round_trips.append({
                        "asset": asset,
                        "entry_date": entry_date,
                        "exit_date": date,
                        "profit": proceeds - cost,
                        "holding_days": (date - entry_date).days,
                    })

        return round_trips


# ---------------------------
# Example Usage
# ---------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-01", periods=200)

    # Two synthetic price series
    df = pd.DataFrame(
        {
            "Asset1": np.cumsum(rng.standard_normal(200)) + 150,
            "Asset2": np.cumsum(rng.standard_normal(200)) + 100,
        },
        index=dates,
    )

    def pct_commission(price: float, units: float) -> float:
        """Return 0.1 % of notional as commission.

        Args:
            price: Execution price.
            units: Number of units traded.

        Returns:
            float: Commission cost.

        """
        return price * units * 0.001

    bt = Backtester(
        df,
        MovingAverageCross,
        initial_cash=10_000,
        commission_fn=pct_commission,
        position_sizing=0.9,
    )
    result = bt.run()
    print(bt.stats())
    result[["Asset1", "Asset2", "Equity"]].plot(title="Equity Curve", secondary_y="Equity")
