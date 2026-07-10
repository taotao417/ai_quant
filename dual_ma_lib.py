"""dual_ma_lib.py —— 双均线（金叉 / 死叉）策略回测核心库

纯 Python + pandas 实现，**不依赖任何回测框架**。
可被命令行脚本 `dual_ma_backtest.py` 与 Jupyter 笔记 `双均线策略回测.ipynb` 复用。

设计要点（对照回测常见坑）：
- 信号用“穿越瞬间”事件（金叉 / 死叉），不是持续状态。
- 信号在 bar i 生成，执行在 bar i+1 开盘价（防前视偏差 / look-ahead）。
- 止盈 / 止损在 bar i 根据当日 high/low/close 检测，于 bar i+1 开盘价执行（同样防前视）。
- A 股：T+1（买入次日才能卖）、100 股整手、印花税仅卖方、佣金双边。
- 基金：ETF/LOF 用券商佣金（无印花税）、100 份整手；场外基金用申购费+赎回费分档、份额可为小数。
- 指标预热（warmup）：均线在前面若干根才有效，预热段只更新状态、不产生交易。
- 末尾强制平仓并计入成交记录，避免权益含浮盈而 trades 缺记录。

成本模型（cost_model）：
- "stock"：佣金双边万 2.5（最低 5 元）+ 印花税千 1（仅卖），100 股整手。
- "etf"  ：券商佣金（默认万 0.5，无最低）+ 无印花税，100 份整手。
- "otc"  ：申购费（默认 0.1%）+ 赎回费分档（<7天 1.5% / 7天~1年 0.5% / ≥1年 0），
          无印花税，份额可为小数（按金额申购）。
"""
from __future__ import annotations

import sys
import numpy as np
import pandas as pd
import matplotlib

# 无界面脚本环境用 Agg；在 Jupyter(ipjykernel) 里保留 inline 后端以正确内嵌图像
if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 中文字体（macOS 优先 PingFang SC，保证图表中文不乱码）
plt.rcParams["font.sans-serif"] = [
    "PingFang SC", "Arial Unicode MS", "Heiti SC", "STHeiti", "SimHei", "DejaVu Sans"
]
plt.rcParams["axes.unicode_minus"] = False


def _show():
    """仅在非 Agg（如 notebook inline）后端才显示；Agg 下跳过以消除噪声。"""
    if matplotlib.get_backend().lower() != "agg":
        plt.show()


# ===========================================================================
# 费用辅助
# ===========================================================================
def _sell_fee_rate(cost_model: str, holding_bars: int, sell_comm: float) -> float:
    """卖出费率。股票/ETF 为常数券商佣金；场外基金按持有期分档赎回费。"""
    if cost_model == "otc":
        if holding_bars < 5:       # 近似 <7 个自然日
            return 0.015
        if holding_bars < 252:     # 近似 7 天 ~ 1 年
            return 0.005
        return 0.0                 # 持有 ≥1 年
    return sell_comm


def _min_comm_eff(cost_model: str, min_comm: float) -> float:
    """基金无最低佣金（或忽略不计），仅股票保留最低 5 元。"""
    return min_comm if cost_model == "stock" else 0.0


def _sell_tax_eff(cost_model: str, sell_tax: float) -> float:
    """仅 A 股收印花税；基金（ETF/场外）无印花税。"""
    return sell_tax if cost_model == "stock" else 0.0


# ===========================================================================
# 1. 数据加载
# ===========================================================================
def load_price_csv(path: str) -> pd.DataFrame:
    """加载已存储的股价数据（至少包含 date, open, high, low, close）。

    date 同时兼容「2023-01-03」与「20230103」两种写法：统一转字符串、去横线后按 %Y%m%d 解析，
    避免整数格式被 pd.to_datetime 误当成「纳秒时间戳」而解析成 1970 年。
    """
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(
        df["date"].astype(str).str.replace("-", "", regex=False), format="%Y%m%d"
    ).dt.strftime("%Y-%m-%d")
    df = df.sort_values("date").reset_index(drop=True)
    for col in ["open", "high", "low", "close", "vol", "amount", "pre_close"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "close" not in df.columns:
        raise ValueError(f"数据缺少 close 列：{path}")
    return df


# ===========================================================================
# 2. 指标：均线 + 金叉 / 死叉事件
# ===========================================================================
def compute_mas(df: pd.DataFrame, short: int, long: int) -> pd.DataFrame:
    """计算短 / 长均线（向量化，SMA）。"""
    df = df.copy()
    df["ma_s"] = df["close"].rolling(short).mean()
    df["ma_l"] = df["close"].rolling(long).mean()
    return df


def compute_cross_signals(df: pd.DataFrame) -> pd.DataFrame:
    """金叉 / 死叉是“穿越瞬间”事件，用 shift(1) 比较，不是持续状态。

    金叉：短均线由下向上穿越长均线（prev_diff<=0 且 diff>0）。
    死叉：短均线由上向下跌破长均线（prev_diff>=0 且 diff<0）。
    """
    df = df.copy()
    diff = df["ma_s"] - df["ma_l"]
    prev_diff = diff.shift(1)
    df["golden_cross"] = (prev_diff <= 0) & (diff > 0)
    df["death_cross"] = (prev_diff >= 0) & (diff < 0)
    return df


# ===========================================================================
# 3. 回测引擎
# ===========================================================================
def run_backtest(
    df: pd.DataFrame,
    short: int = 5,
    long: int = 15,
    initial_cash: float = 100_000.0,
    buy_comm: float = 0.00025,      # 买入费率（股票/ETF=券商佣金；场外=申购费）
    sell_comm: float = 0.00025,     # 卖出基础费率（股票/ETF=券商佣金；场外忽略，用赎回分档）
    sell_tax: float = 0.001,        # 印花税（仅 A 股生效）
    slippage: float = 0.0,          # 滑点：买入 +slippage，卖出 -slippage
    lot: int = 100,                 # 整手（股票/ETF=100 股；场外传 1 含义为份额单位）
    min_comm: float = 5.0,          # 最低佣金（仅股票生效）
    start_date: str | None = None,  # 评估区间起点（用户可指定）
    end_date: str | None = None,    # 评估区间终点
    symbol: str = "",
    symbol_name: str = "",
    cost_model: str = "stock",      # "stock" | "etf" | "otc"
    fractional: bool = False,       # 场外基金份额可为小数
    stop_loss: float | None = None,     # 止损比例，如 0.08 = 8%（相对买入价）
    take_profit: float | None = None,   # 止盈比例，如 0.20 = 20%（相对买入价）
) -> tuple[list[dict], list[dict], dict]:
    """运行双均线回测（含可选止盈 / 止损）。

    返回 (equity_curve, trade_history, meta)。
    - equity_curve: [{date, value}]，仅评估窗口内。
    - trade_history: 已平仓成交（含末尾强制平仓），含 exit_reason（cross/sl/tp/end）。
    - meta: 评估窗口起止、参数等。
    """
    if short >= long:
        raise ValueError("短均线周期必须 < 长均线周期")

    df = df.copy()
    df["ma_s"] = df["close"].rolling(short).mean()
    df["ma_l"] = df["close"].rolling(long).mean()

    n = len(df)
    warmup_idx = long
    if start_date is not None:
        user_idx = df.index[df["date"] >= str(start_date)[:10]]
        if len(user_idx):
            warmup_idx = max(warmup_idx, int(user_idx[0]))
    end_idx = n - 1
    if end_date is not None:
        eidx = df.index[df["date"] <= str(end_date)[:10]]
        if len(eidx):
            end_idx = int(eidx[-1])
    if warmup_idx >= end_idx:
        raise ValueError("评估区间过短，无法产生有效信号（长均线周期过大或区间太短）")

    min_comm_eff = _min_comm_eff(cost_model, min_comm)
    sell_tax_eff = _sell_tax_eff(cost_model, sell_tax)

    equity_curve: list[dict] = []
    trade_history: list[dict] = []
    cash = float(initial_cash)
    position = 0.0
    entry_price = 0.0
    entry_comm = 0.0
    entry_date = None
    entry_idx = None
    pending_buy = False
    pending_sell = False
    pending_exit = None  # "sl" / "tp"

    for i in range(warmup_idx, end_idx + 1):
        row = df.iloc[i]
        date = str(row["date"])[:10]
        o = float(row["open"])
        c = float(row["close"])
        ma_s = row["ma_s"]
        ma_l = row["ma_l"]
        lo = float(row["low"])
        hi = float(row["high"])

        # (1) 先执行昨日挂单：用“次日开盘价”，防前视
        if pending_buy and position == 0 and not pd.isna(ma_s):
            px = o * (1.0 + slippage)
            if fractional:
                size = cash / (px * (1.0 + buy_comm))
            else:
                size = int(cash / (px * (1.0 + buy_comm)) // lot) * lot
            if size > 0:
                comm = max(px * size * buy_comm, min_comm_eff)
                cash -= px * size + comm
                position = size
                entry_price = px
                entry_comm = comm
                entry_date = date
                entry_idx = i
                pending_buy = False

        if (pending_sell or pending_exit) and position > 0:
            px = o * (1.0 - slippage)
            hb = int(i - entry_idx)
            sfee = _sell_fee_rate(cost_model, hb, sell_comm) + sell_tax_eff
            comm = max(px * position * sfee, min_comm_eff)
            proceeds = px * position * (1.0 - sfee)
            cash += proceeds
            cost_basis = entry_price * position + entry_comm
            pnl = proceeds - cost_basis
            reason = pending_exit if pending_exit else "cross"
            trade_history.append({
                "entry_date": entry_date,
                "exit_date": date,
                "side": "long",
                "size": round(position, 4) if fractional else int(position),
                "entry_price": round(entry_price, 4),
                "exit_price": round(px, 4),
                "pnl": round(pnl, 2),
                "pnl_pct": round(pnl / cost_basis * 100.0, 2),
                "holding_bars": hb,
                "exit_reason": reason,
                "symbol": symbol,
                "symbol_name": symbol_name,
            })
            position = 0.0
            pending_sell = False
            pending_exit = None

        # (2) 生成今日信号（给下一根执行）
        prev_s = df.iloc[i - 1]["ma_s"]
        prev_l = df.iloc[i - 1]["ma_l"]
        if not pd.isna(ma_s) and not pd.isna(ma_l) and not pd.isna(prev_s) and not pd.isna(prev_l):
            diff = ma_s - ma_l
            prev_diff = prev_s - prev_l
            if prev_diff <= 0 and diff > 0 and position == 0:
                pending_buy = True
            if prev_diff >= 0 and diff < 0 and position > 0 and not pending_exit:
                pending_sell = True

        # (2b) 止盈 / 止损检测（持有中，bar i 触发，bar i+1 开盘价执行）
        if position > 0 and not pending_exit:
            if take_profit is not None and (hi >= entry_price * (1.0 + take_profit)
                                            or c >= entry_price * (1.0 + take_profit)):
                pending_exit = "tp"
            elif stop_loss is not None and (lo <= entry_price * (1.0 - stop_loss)
                                           or c <= entry_price * (1.0 - stop_loss)):
                pending_exit = "sl"

        # (3) 记录权益（收盘市值）
        equity_curve.append({"date": date, "value": round(cash + position * c, 2)})

    # 末尾强制平仓（样本结束仍有持仓）
    if position > 0:
        last = df.iloc[end_idx]
        date = str(last["date"])[:10]
        px = float(last["close"]) * (1.0 - slippage)
        hb = int(end_idx - entry_idx)
        sfee = _sell_fee_rate(cost_model, hb, sell_comm) + sell_tax_eff
        comm = max(px * position * sfee, min_comm_eff)
        proceeds = px * position * (1.0 - sfee)
        cash += proceeds
        cost_basis = entry_price * position + entry_comm
        pnl = proceeds - cost_basis
        trade_history.append({
            "entry_date": entry_date,
            "exit_date": date,
            "side": "long",
            "size": round(position, 4) if fractional else int(position),
            "entry_price": round(entry_price, 4),
            "exit_price": round(px, 4),
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl / cost_basis * 100.0, 2),
            "holding_bars": hb,
            "exit_reason": "end",
            "symbol": symbol,
            "symbol_name": symbol_name,
        })
        position = 0.0
        equity_curve[-1]["value"] = round(cash, 2)

    meta = {
        "symbol": symbol,
        "symbol_name": symbol_name,
        "short": short,
        "long": long,
        "initial_cash": initial_cash,
        "cost_model": cost_model,
        "buy_comm": buy_comm,
        "sell_comm": sell_comm,
        "sell_tax": sell_tax,
        "slippage": slippage,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "fractional": fractional,
        "eval_start": equity_curve[0]["date"] if equity_curve else None,
        "eval_end": equity_curve[-1]["date"] if equity_curve else None,
        "is_flat_at_end": position == 0,
    }
    return equity_curve, trade_history, meta


# ===========================================================================
# 4. 买入持有基准
# ===========================================================================
def buy_and_hold(df: pd.DataFrame, initial_cash: float = 100_000.0,
                 start_idx: int = 0, end_idx: int | None = None,
                 buy_comm: float = 0.00025, sell_comm: float = 0.00025,
                 sell_tax: float = 0.001, slippage: float = 0.0,
                 lot: int = 100, min_comm: float = 5.0,
                 cost_model: str = "stock", fractional: bool = False) -> list[dict]:
    """同区间买入持有，作为策略对照基准（含同等费用）。"""
    if end_idx is None:
        end_idx = len(df) - 1
    o0 = float(df.iloc[start_idx]["open"]) * (1.0 + slippage)
    min_comm_eff = _min_comm_eff(cost_model, min_comm)
    sell_tax_eff = _sell_tax_eff(cost_model, sell_tax)
    if fractional:
        size = initial_cash / (o0 * (1.0 + buy_comm))
    else:
        size = int(initial_cash / (o0 * (1.0 + buy_comm)) // lot) * lot
    comm0 = max(o0 * size * buy_comm, min_comm_eff)
    cash = initial_cash - (o0 * size + comm0)
    eq = []
    for i in range(start_idx, end_idx + 1):
        c = float(df.iloc[i]["close"])
        eq.append({"date": str(df.iloc[i]["date"])[:10], "value": round(cash + size * c, 2)})
    ce = float(df.iloc[end_idx]["close"]) * (1.0 - slippage)
    hb = end_idx - start_idx
    sfee = _sell_fee_rate(cost_model, hb, sell_comm) + sell_tax_eff
    comm1 = max(ce * size * sfee, min_comm_eff)
    cash += ce * size * (1.0 - sfee)
    eq[-1]["value"] = round(cash, 2)
    return eq


# ===========================================================================
# 5. 指标计算（与 export_results 口径一致，用于扫描 / 展示）
# ===========================================================================
def compute_metrics(equity_curve: list[dict], trade_history: list[dict]) -> dict:
    vals = [p["value"] for p in equity_curve]
    if not vals:
        return {}
    base, final = vals[0], vals[-1]
    total_return = (final / base - 1.0) * 100.0
    rets = np.diff(vals) / np.array(vals[:-1])
    n = len(rets)
    annual = ((final / base) ** (252.0 / n) - 1.0) * 100.0 if (n > 0 and final / base > 0) else None
    sharpe = float(rets.mean() / rets.std() * np.sqrt(252.0)) if (n > 1 and rets.std() > 0) else None
    peak = np.maximum.accumulate(vals)
    dd = (np.array(vals) / peak - 1.0) * 100.0
    mdd = float(abs(dd.min()))
    wins = sum(1 for t in trade_history if t["pnl"] > 0)
    win_rate = wins / len(trade_history) * 100.0 if trade_history else 0.0
    reasons = {}
    for t in trade_history:
        reasons[t.get("exit_reason", "unknown")] = reasons.get(t.get("exit_reason", "unknown"), 0) + 1
    return {
        "total_return_pct": round(total_return, 2),
        "annual_return_pct": round(annual, 2) if annual is not None else None,
        "sharpe": round(sharpe, 3) if sharpe is not None else None,
        "max_drawdown_pct": round(mdd, 2),
        "win_rate_pct": round(win_rate, 2),
        "total_trades": len(trade_history),
        "exit_reasons": reasons,
    }


# ===========================================================================
# 6. 可视化
# ===========================================================================
def _xdates(dates):
    return pd.to_datetime(dates)


def plot_price_signals(df: pd.DataFrame, short: int, long: int,
                        trade_history: list[dict], symbol_name: str = "",
                        save: str | None = None) -> plt.Figure:
    """价格 + 短/长均线 + 买入(▲)/卖出(▼)信号标记。"""
    fig, ax = plt.subplots(figsize=(15, 6))
    xd = _xdates(df["date"])
    ax.plot(xd, df["close"], label="收盘价", lw=1.2, color="#222")
    ax.plot(xd, df[f"ma_s"] if "ma_s" in df else df["close"].rolling(short).mean(),
            label=f"MA{short}（短均线）", lw=1.3)
    ax.plot(xd, df[f"ma_l"] if "ma_l" in df else df["close"].rolling(long).mean(),
            label=f"MA{long}（长均线）", lw=1.3)

    for t in trade_history:
        ax.scatter(pd.Timestamp(t["entry_date"]), t["entry_price"], marker="^",
                   color="#1a9850", s=110, zorder=5, label="买入" if "买入" not in ax.get_legend_handles_labels()[1] else "")
        ax.scatter(pd.Timestamp(t["exit_date"]), t["exit_price"], marker="v",
                   color="#d73027", s=110, zorder=5, label="卖出" if "卖出" not in ax.get_legend_handles_labels()[1] else "")

    ax.set_title(f"{symbol_name} 双均线策略（MA{short} / MA{long}）：价格、均线与买卖信号", fontsize=13)
    ax.set_ylabel("价格（元）")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.legend(loc="best")
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    if save:
        fig.savefig(save, dpi=110, bbox_inches="tight")
    _show()
    return fig


def plot_equity(strat_eq: list[dict], bh_eq: list[dict], symbol_name: str = "",
                save: str | None = None) -> plt.Figure:
    """策略资金曲线 vs 买入持有基准。"""
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(_xdates([p["date"] for p in strat_eq]), [p["value"] for p in strat_eq],
            label="双均线策略", lw=1.6, color="#1f77b4")
    ax.plot(_xdates([p["date"] for p in bh_eq]), [p["value"] for p in bh_eq],
            label="买入持有", lw=1.4, color="#999", ls="--")
    ax.set_title(f"{symbol_name} 资金曲线：策略 vs 买入持有", fontsize=13)
    ax.set_ylabel("账户净值（元）")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.legend(loc="best")
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    if save:
        fig.savefig(save, dpi=110, bbox_inches="tight")
    _show()
    return fig


def plot_drawdown(equity_curve: list[dict], symbol_name: str = "",
                  save: str | None = None) -> plt.Figure:
    """回撤曲线（最大回撤可视化）。"""
    vals = np.array([p["value"] for p in equity_curve])
    peak = np.maximum.accumulate(vals)
    dd = (vals / peak - 1.0) * 100.0
    fig, ax = plt.subplots(figsize=(15, 3.6))
    ax.fill_between(_xdates([p["date"] for p in equity_curve]), dd, 0, color="#d73027", alpha=0.35)
    ax.plot(_xdates([p["date"] for p in equity_curve]), dd, color="#d73027", lw=1.0)
    ax.set_title(f"{symbol_name} 回撤曲线（最大回撤 {abs(dd.min()):.2f}%）", fontsize=13)
    ax.set_ylabel("回撤 %")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    if save:
        fig.savefig(save, dpi=110, bbox_inches="tight")
    _show()
    return fig


def plot_sweep_heatmap(grid_result: pd.DataFrame, symbol_name: str = "",
                       save: str | None = None) -> plt.Figure:
    """参数扫描热力图：横轴 short，纵轴 long，颜色为累计收益率%。"""
    pivot = grid_result.pivot(index="long", columns="short", values="total_return_pct")
    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto",
                   vmin=np.nanmin(pivot.values), vmax=np.nanmax(pivot.values))
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("短均线周期")
    ax.set_ylabel("长均线周期")
    ax.set_title(f"{symbol_name} 双均线参数扫描：累计收益率 (%)", fontsize=13)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="累计收益 %")
    if save:
        fig.savefig(save, dpi=110, bbox_inches="tight")
    _show()
    return fig


def tp_sl_grid(df: pd.DataFrame, short: int, long: int,
               sl_list: list[float], tp_list: list[float],
               cost_model: str = "stock", fractional: bool = False,
               **kw) -> pd.DataFrame:
    """止盈 / 止损网格扫描：对每个 (止损, 止盈) 组合跑回测，返回指标表。"""
    rows = []
    for sl in sl_list:
        for tp in tp_list:
            eq, tr, _ = run_backtest(
                df, short=short, long=long, cost_model=cost_model,
                fractional=fractional, stop_loss=sl, take_profit=tp, **kw)
            m = compute_metrics(eq, tr)
            rows.append({
                "stop_loss": sl, "take_profit": tp,
                "total_return_pct": m["total_return_pct"],
                "sharpe": m["sharpe"],
                "max_drawdown_pct": m["max_drawdown_pct"],
                "win_rate_pct": m["win_rate_pct"],
                "total_trades": m["total_trades"],
            })
    return pd.DataFrame(rows)


def plot_tp_sl_heatmap(grid: pd.DataFrame, value_col: str = "total_return_pct",
                       symbol_name: str = "", save: str | None = None) -> plt.Figure:
    """TP/SL 网格热力图：行=止损，列=止盈，颜色=value_col。"""
    pivot = grid.pivot(index="stop_loss", columns="take_profit", values=value_col)
    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto",
                   vmin=np.nanmin(pivot.values), vmax=np.nanmax(pivot.values))
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{v:.0%}" for v in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{v:.0%}" for v in pivot.index])
    ax.set_xlabel("止盈比例")
    ax.set_ylabel("止损比例")
    ax.set_title(f"{symbol_name} 止盈/止损网格：{value_col}", fontsize=13)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label=value_col)
    if save:
        fig.savefig(save, dpi=110, bbox_inches="tight")
    _show()
    return fig
