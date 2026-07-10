"""dual_ma_backtest.py —— 双均线策略回测（命令行可运行版）

用法：
    python dual_ma_backtest.py

产出（当前目录）：
    dual_ma_002353_equity.csv / _trades.csv / _summary.json   —— 标准三件套
    dual_ma_price_signals.png / _equity.png / _drawdown.png / _sweep.png —— 图表

策略：短均线由下向上穿越长均线（金叉）买入，由上向下跌破（死叉）卖出。
执行：信号次日开盘价成交（防前视）；A 股 T+1、100 股整手、印花税仅卖方。
"""
from __future__ import annotations

import json
import os

import dual_ma_lib as lib
from export_results import export_results

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "杰瑞股份_日线_前复权.csv")
PREFIX = "dual_ma_002353"
SYMBOL = "002353.SZ"
SYMBOL_NAME = "杰瑞股份"


def main():
    df = lib.load_price_csv(DATA_PATH)
    print(f"已加载 {SYMBOL_NAME} 数据：{len(df)} 行，"
          f"{df['date'].iloc[0]} ~ {df['date'].iloc[-1]}")

    short, long = 5, 15
    equity_curve, trade_history, meta = lib.run_backtest(
        df, short=short, long=long, initial_cash=100_000.0,
        buy_comm=0.00025, sell_comm=0.00025, sell_tax=0.001,
        slippage=0.0, lot=100, min_comm=5.0,
        symbol=SYMBOL, symbol_name=SYMBOL_NAME,
    )

    # ---- 标准三件套 ----
    paths = export_results(
        equity_curve=equity_curve,
        trade_history=trade_history,
        prefix=PREFIX,
        initial_cash=meta["initial_cash"],
        start=meta["eval_start"],
        end=meta["eval_end"],
        market="china_a",
        strategy_name="双均线(金叉/死叉)",
        symbol=SYMBOL,
        is_flat_at_end=meta["is_flat_at_end"],
    )
    print("已写出标准文件：", {k: str(v) for k, v in paths.items()})

    # ---- 买入持有基准 ----
    s_idx = df.index[df["date"] == meta["eval_start"]][0]
    e_idx = df.index[df["date"] == meta["eval_end"]][0]
    bh_eq = lib.buy_and_hold(df, initial_cash=100_000.0, start_idx=s_idx, end_idx=e_idx)

    # ---- 指标 ----
    metrics = lib.compute_metrics(equity_curve, trade_history)
    bh_vals = [p["value"] for p in bh_eq]
    bh_ret = (bh_vals[-1] / bh_vals[0] - 1) * 100
    print("\n==== 回测结果（评估区间 "
          f"{meta['eval_start']} ~ {meta['eval_end']}）====")
    for k, v in metrics.items():
        print(f"  {k:18s}: {v}")
    print(f"  买入持有累计收益  : {bh_ret:.2f}%")
    print(f"  信号次数（买入）  : {sum(1 for t in trade_history)} 笔完整交易")

    # ---- 图表 ----
    df_ma = lib.compute_mas(df, short, long)
    lib.plot_price_signals(df_ma, short, long, trade_history, symbol_name=SYMBOL_NAME,
                           save=os.path.join(HERE, f"{PREFIX}_price_signals.png"))
    lib.plot_equity(equity_curve, bh_eq, symbol_name=SYMBOL_NAME,
                    save=os.path.join(HERE, f"{PREFIX}_equity.png"))
    lib.plot_drawdown(equity_curve, symbol_name=SYMBOL_NAME,
                      save=os.path.join(HERE, f"{PREFIX}_drawdown.png"))

    # ---- 参数扫描（不同均线周期）----
    shorts = [3, 5, 8, 10, 15, 20]
    longs = [10, 15, 20, 30, 40, 60]
    rows = []
    for sh in shorts:
        for lo in longs:
            if sh >= lo:
                continue
            eq, tr, _ = lib.run_backtest(df, short=sh, long=lo,
                                        initial_cash=100_000.0, symbol=SYMBOL,
                                        symbol_name=SYMBOL_NAME)
            m = lib.compute_metrics(eq, tr)
            rows.append({"short": sh, "long": lo,
                         "total_return_pct": m["total_return_pct"],
                         "sharpe": m["sharpe"],
                         "max_drawdown_pct": m["max_drawdown_pct"],
                         "total_trades": m["total_trades"]})
    grid = pd_DataFrame(rows)
    grid.to_csv(os.path.join(HERE, f"{PREFIX}_sweep.csv"), index=False)
    lib.plot_sweep_heatmap(grid, symbol_name=SYMBOL_NAME,
                           save=os.path.join(HERE, f"{PREFIX}_sweep.png"))
    print(f"\n参数扫描完成，已写出 {PREFIX}_sweep.csv / .png")

    # summary 也留一份易读拷贝
    with open(os.path.join(HERE, f"{PREFIX}_metrics.txt"), "w", encoding="utf-8") as f:
        f.write(json.dumps({"metrics": metrics, "buy_and_hold_return_pct": round(bh_ret, 2)},
                           ensure_ascii=False, indent=2))


def pd_DataFrame(rows):
    import pandas as pd
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
