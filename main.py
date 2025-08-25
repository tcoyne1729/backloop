import numpy as np
import pandas as pd


# ---------------------------
# Strategy Base Class
# ---------------------------
class Strategy:
    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()

    def generate_signals(self) -> pd.Series:
        """
        Must be implemented in child classes.
        Should return a pd.Series with values: 1 (long), 0 (flat), -1 (short).
        """
        raise NotImplementedError("Strategy must implement generate_signals()")


# Example Strategy: Moving Average Crossover
class MovingAverageCross(Strategy):
    def __init__(self, data, short_window=20, long_window=50):
        super().__init__(data)
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self):
        short_ma = self.data["Close"].rolling(self.short_window).mean()
        long_ma = self.data["Close"].rolling(self.long_window).mean()
        signals = (short_ma > long_ma).astype(int)  # 1 = long, 0 = flat
        return signals


# ---------------------------
# Backtester
# ---------------------------
class Backtester:
    def __init__(self, data: pd.DataFrame, strategy: Strategy,
                 initial_cash=10_000, commission_fn=None,
                 position_sizing=1.0):
        """
        commission_fn: function(price, size) -> commission_cost
                       If None, no commission is applied.
        position_sizing: fraction of available cash to allocate (0.0 - 1.0)
        """
        self.data = data.copy()
        self.strategy = strategy(data)
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.position = 0
        self.units = 0
        self.trades = []  # (entry_price, exit_price, profit)
        self.equity_curve = []
        self.commission_fn = commission_fn if commission_fn else self.default_commission
        self.position_sizing = position_sizing

    def default_commission(self, price, size):
        """Default commission function is no commission.

        Args:
            price (float): price of instrument
            size (float): notional size of trade

        Returns:
            float: commission (zero for default but can be overridden in class instatiation).

        """
        return 0.0

    def run(self):
        signals = self.strategy.generate_signals()
        self.data["Signal"] = signals

        entry_price = None

        for _, row in self.data.iterrows():
            price = row["Close"]
            signal = row["Signal"]

            # Enter long position
            if self.position == 0 and signal == 1:
                alloc_cash = self.cash * self.position_sizing
                self.units = alloc_cash // price  # integer shares
                if self.units > 0:
                    entry_price = price
                    cost = self.units * price
                    commission = self.commission_fn(price, self.units)
                    self.cash -= (cost + commission)
                    self.position = 1

            # Exit position
            elif self.position == 1 and signal == 0:
                proceeds = self.units * price
                commission = self.commission_fn(price, self.units)
                self.cash += proceeds - commission
                profit = proceeds - commission - (entry_price * self.units)
                self.trades.append((entry_price, price, profit, self.units))
                entry_price = None
                self.position = 0
                self.units = 0

            # Track equity
            if self.position == 1:
                equity = self.cash + self.units * price
            else:
                equity = self.cash

            self.equity_curve.append(equity)

        self.data["Equity"] = self.equity_curve
        return self.data

    def stats(self) -> dict:
        """Return basic stats of the strategy.

        Returns:
            dict: dictionary of output metrics.

        """
        df = pd.DataFrame(self.trades, columns=["Entry", "Exit", "Profit", "Units"])
        total_profit = df["Profit"].sum() if not df.empty else 0
        win_rate = (df["Profit"] > 0).mean() if not df.empty else np.nan
        max_dd = self._max_drawdown()

        return {
            "Final Equity": self.equity_curve[-1] if self.equity_curve else self.initial_cash,
            "Total Profit": total_profit,
            "Win Rate": win_rate,
            "Max Drawdown": max_dd
        }

    def _max_drawdown(self):
        equity = np.array(self.equity_curve)
        if len(equity) == 0:
            return 0
        peak = np.maximum.accumulate(equity)
        dd = (equity - peak) / peak
        return dd.min()

    print("Hello from curve-fitting!")


# ---------------------------
# Example Usage
# ---------------------------
if __name__ == "__main__":
    # Dummy price series
    dates = pd.date_range("2020-01-01", periods=200)
    prices = np.cumsum(np.random.randn(200)) + 100
    df = pd.DataFrame({"Close": prices}, index=dates)


    # Commission function: 0.1% of trade value
    def pct_commission(price: float, size: float) -> float:
        """Return the commission of the trade as a percentage of the notional.

        Args:
            price (float): price trade executed at
            size (float): size of the trade

        Returns:
            float: commission of the trade in the trade currency

        """
        return price * size * 0.001


    # Run with Moving Average Cross strategy, 50% of capital each trade
    bt = Backtester(
        df,
        MovingAverageCross,
        initial_cash=10000,
        commission_fn=pct_commission,
        position_sizing=0.5
    )
    result = bt.run()
    print(bt.stats())
    result[["Close", "Equity"]].plot(title="Equity Curve")
