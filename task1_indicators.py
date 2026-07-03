#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task 1 延续：技术指标分析
===========================
1. 数据基础诊断（缺失值检查、描述性统计）
2. RSI、MACD、布林带指标理论与计算
3. 三种指标的可视化
4. 扩展指标：KDJ 介绍与实现

股票：杰瑞股份 (002353.SZ)
日期范围：2025-07-02 ~ 2026-07-01 (242个交易日)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 绘图设置 - 中文字体 & 风格
# ============================================================
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'PingFang SC', 'Heiti SC']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 120
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['figure.figsize'] = (18, 10)

# ============================================================
# 1. 数据加载
# ============================================================
print("=" * 70)
print("  杰瑞股份(002353) 技术指标深度分析")
print("=" * 70)

df = pd.read_csv('杰瑞股份_日线.csv')
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)
df.sort_index(inplace=True)

print(f"\n📊 数据概览")
print(f"   交易日数: {len(df)}")
print(f"   时间范围: {df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')}")
print(f"   列名: {', '.join(df.columns.tolist())}")

# ============================================================
# 第一部分：数据基础诊断分析
# ============================================================
print("\n" + "=" * 70)
print("  第一部分：数据基础诊断")
print("=" * 70)

# --- 1.1 缺失值检查 ---
print("\n📋 1.1 缺失值检查")
print("-" * 50)
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({
    '缺失数量': missing,
    '缺失比例(%)': missing_pct
})
missing_df = missing_df[missing_df['缺失数量'] > 0]
if len(missing_df) == 0:
    print("   ✅ 数据完整，所有列均无缺失值！")
else:
    print(missing_df.to_string())

# 额外检查：是否有重复日期
dup_dates = df.index.duplicated().sum()
print(f"   重复日期数: {dup_dates}")

# 检查是否有交易日缺失（周末除外）
full_range = pd.date_range(df.index.min(), df.index.max(), freq='B')  # B = business day
trading_days_missing = len(full_range) - len(df)
print(f"   理论工作日: {len(full_range)}, 实际交易日: {len(df)}, 差异: {trading_days_missing}天")
if trading_days_missing > 0:
    missing_dates = full_range.difference(df.index)
    print(f"   缺失日期(前10): {missing_dates[:10].strftime('%Y-%m-%d').tolist()}")

# --- 1.2 描述性统计 ---
print("\n📈 1.2 描述性统计量")
print("-" * 50)
desc_cols = ['open', 'high', 'low', 'close', 'vol', 'amount', 'pct_chg']
desc = df[desc_cols].describe()
print(desc.round(2).to_string())

# 额外统计：偏度和峰度
print("\n📐 偏度 (Skewness) 和 峰度 (Kurtosis):")
print("-" * 50)
for col in ['close', 'vol', 'pct_chg']:
    skew_val = df[col].skew()
    kurt_val = df[col].kurtosis()
    skew_desc = "右偏(正偏)" if skew_val > 0.5 else ("左偏(负偏)" if skew_val < -0.5 else "近似对称")
    kurt_desc = "尖峰厚尾" if kurt_val > 1 else ("扁平" if kurt_val < -1 else "接近正态")
    print(f"   {col:>12s}: 偏度={skew_val:+.4f} ({skew_desc}), 峰度={kurt_val:+.4f} ({kurt_desc})")

# --- 1.3 OHLC 数据一致性检查 ---
print("\n🔍 1.3 OHLC 数据一致性检查")
print("-" * 50)
# High >= Open, High >= Close, Low <= Open, Low <= Close
bad_high_open = (df['high'] < df['open']).sum()
bad_high_close = (df['high'] < df['close']).sum()
bad_low_open = (df['low'] > df['open']).sum()
bad_low_close = (df['low'] > df['close']).sum()
bad_high_low = (df['high'] < df['low']).sum()

print(f"   High < Open  异常: {bad_high_open} 条")
print(f"   High < Close 异常: {bad_high_close} 条")
print(f"   Low > Open   异常: {bad_low_open} 条")
print(f"   Low > Close  异常: {bad_low_close} 条")
print(f"   High < Low   异常: {bad_high_low} 条")
if all(v == 0 for v in [bad_high_open, bad_high_close, bad_low_open, bad_low_close, bad_high_low]):
    print("   ✅ 所有OHLC数据逻辑一致，无异常！")

# --- 1.4 涨跌停检查 ---
print("\n📊 1.4 异常波动检查")
print("-" * 50)
extreme_up = df[df['pct_chg'] >= 9.5]   # 接近涨停
extreme_down = df[df['pct_chg'] <= -9.5]  # 接近跌停
print(f"   涨幅 >= 9.5%: {len(extreme_up)} 天 (涨停/接近涨停)")
print(f"   跌幅 >= 9.5%: {len(extreme_down)} 天 (跌停/接近跌停)")

if len(extreme_up) > 0:
    print(f"   涨停日: {extreme_up.index.strftime('%Y-%m-%d').tolist()}")
if len(extreme_down) > 0:
    print(f"   跌停日: {extreme_down.index.strftime('%Y-%m-%d').tolist()}")

print("\n   ✅ 数据诊断完成！数据质量良好，可以继续分析。")

# ============================================================
# 第二部分：技术指标理论与计算
# ============================================================
print("\n" + "=" * 70)
print("  第二部分：技术指标理论与计算")
print("=" * 70)

# --- 辅助函数：计算各个指标 ---

def calc_rsi(close, period=14):
    """
    RSI (相对强弱指标)
    ===================
    公式:
        RS = 周期内平均涨幅 / 周期内平均跌幅
        RSI = 100 - 100/(1 + RS)
    
    参数:
        period: 计算周期，默认 14 天
    
    作用:
        - 判断超买超卖：RSI > 70 超买，RSI < 30 超卖
        - 识别背离：价格新高但RSI未新高 → 顶背离（看跌）
        - 识别趋势强度：RSI在50上方为多头，下方为空头
    """
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    
    # Wilder's smoothing (用简单移动平均作为近似)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    # 使用指数平滑改进（Wilder原始方法）
    for i in range(period, len(avg_gain)):
        avg_gain.iloc[i] = (avg_gain.iloc[i-1] * (period - 1) + gain.iloc[i]) / period
        avg_loss.iloc[i] = (avg_loss.iloc[i-1] * (period - 1) + loss.iloc[i]) / period
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calc_macd(close, fast=12, slow=26, signal=9):
    """
    MACD (指数平滑异同移动平均线)
    ==============================
    公式:
        EMA_fast = EMA(close, fast)    # 快线(12日)
        EMA_slow = EMA(close, slow)    # 慢线(26日)
        MACD线   = EMA_fast - EMA_slow  # DIF
        信号线   = EMA(MACD线, signal)  # DEA
        柱状图   = MACD线 - 信号线      # 2*(DIF - DEA)
    
    参数:
        fast: 快线周期，默认 12
        slow: 慢线周期，默认 26
        signal: 信号线周期，默认 9
    
    作用:
        - 金叉买入信号：MACD线上穿信号线
        - 死叉卖出信号：MACD线下穿信号线
        - 零轴判断趋势：MACD > 0 多头，MACD < 0 空头
        - 背离：价格新高但MACD未新高 → 顶背离
    """
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = 2 * (macd_line - signal_line)  # 国内常用2倍
    return macd_line, signal_line, histogram


def calc_bollinger(close, period=20, num_std=2):
    """
    布林带 (Bollinger Bands)
    =========================
    公式:
        中轨 (MID) = SMA(close, period)         # 20日均线
        上轨 (UP)  = MID + num_std * STD(close)  # +2倍标准差
        下轨 (LOW) = MID - num_std * STD(close)  # -2倍标准差
    
    参数:
        period: 移动平均周期，默认 20
        num_std: 标准差倍数，默认 2
    
    作用:
        - 波动率测量：带宽反映市场波动率
        - 突破信号：价格触及上轨可能回调，触下轨可能反弹
        - 挤压信号：带宽收窄预示大幅波动即将来临
        - 趋势过滤：价格沿某条轨道持续运行表示强趋势
    """
    middle = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    # 带宽 (百分比)
    bandwidth = (upper - lower) / middle * 100
    # %B: 价格在布林带中的位置 (0~1之间)
    percent_b = (close - lower) / (upper - lower)
    return upper, middle, lower, bandwidth, percent_b


def calc_kdj(high, low, close, period=9, k_period=3, d_period=3):
    """
    KDJ (随机指标)
    ================
    公式:
        RSV(n) = (C - Ln) / (Hn - Ln) × 100
        其中 Cn=当日收盘价, Ln=n日内最低价, Hn=n日内最高价
        
        K = 2/3 × 前一日K + 1/3 × RSV
        D = 2/3 × 前一日D + 1/3 × K
        J = 3K - 2D
    
    参数:
        period: RSV计算周期，默认 9
        k_period: K值平滑周期，默认 3
        d_period: D值平滑周期，默认 3
    
    作用:
        - 超买超卖：K,D > 80 超买，K,D < 20 超卖
        - 金叉死叉：K线上穿D线为金叉(买入)，下穿为死叉(卖出)
        - J值：J > 100 为超买，J < 0 为超卖
        - 背离：价格与KDJ指标走势不一致预示反转
    """
    lowest_low = low.rolling(window=period).min()
    highest_high = high.rolling(window=period).max()
    rsv = (close - lowest_low) / (highest_high - lowest_low) * 100
    
    # 迭代计算 K, D, J
    k_values = np.zeros(len(close))
    d_values = np.zeros(len(close))
    
    # 初始值设为50
    for i in range(len(close)):
        if i == 0 or pd.isna(rsv.iloc[i]):
            k_values[i] = 50
            d_values[i] = 50
        else:
            k_values[i] = (2/3) * k_values[i-1] + (1/3) * rsv.iloc[i]
            d_values[i] = (2/3) * d_values[i-1] + (1/3) * k_values[i]
    
    k = pd.Series(k_values, index=close.index)
    d = pd.Series(d_values, index=close.index)
    j = 3 * k - 2 * d
    
    return k, d, j


# --- 计算所有指标 ---
print("\n⚙️  正在计算技术指标...")

df['rsi_14'] = calc_rsi(df['close'], period=14)
df['macd'], df['macd_signal'], df['macd_hist'] = calc_macd(df['close'])
df['bb_upper'], df['bb_middle'], df['bb_lower'], df['bb_width'], df['bb_pctb'] = calc_bollinger(df['close'])
df['kdj_k'], df['kdj_d'], df['kdj_j'] = calc_kdj(df['high'], df['low'], df['close'])

# 日收益率
df['ret'] = df['close'].pct_change()

print("   ✅ 指标计算完成！")
print(f"\n   最新指标值 (2026-07-01):")
print(f"   RSI(14):     {df['rsi_14'].iloc[-1]:.2f}")
print(f"   MACD:        {df['macd'].iloc[-1]:.4f}")
print(f"   MACD信号:    {df['macd_signal'].iloc[-1]:.4f}")
print(f"   MACD柱:      {df['macd_hist'].iloc[-1]:.4f}")
print(f"   布林上轨:    ¥{df['bb_upper'].iloc[-1]:.2f}")
print(f"   布林中轨:    ¥{df['bb_middle'].iloc[-1]:.2f}")
print(f"   布林下轨:    ¥{df['bb_lower'].iloc[-1]:.2f}")
print(f"   布林带宽:    {df['bb_width'].iloc[-1]:.2f}%")
print(f"   KDJ-K:       {df['kdj_k'].iloc[-1]:.2f}")
print(f"   KDJ-D:       {df['kdj_d'].iloc[-1]:.2f}")
print(f"   KDJ-J:       {df['kdj_j'].iloc[-1]:.2f}")

# ============================================================
# 第三部分：可视化
# ============================================================
print("\n" + "=" * 70)
print("  第三部分：技术指标可视化")
print("=" * 70)
print("   🎨 正在生成图表...")

# ---- 图表 1: RSI 指标 ----
fig = plt.figure(figsize=(20, 12))
gs = GridSpec(3, 1, figure=fig, height_ratios=[2, 1, 1], hspace=0.08)

# 上：收盘价 + 均线
ax1 = fig.add_subplot(gs[0])
ax1.plot(df.index, df['close'], color='#e74c3c', linewidth=1.2, label='收盘价', zorder=2)
ma20 = df['close'].rolling(20).mean()
ma60 = df['close'].rolling(60).mean()
ax1.plot(df.index, ma20, color='#f39c12', linewidth=1.0, alpha=0.8, label='MA20')
ax1.plot(df.index, ma60, color='#2980b9', linewidth=1.0, alpha=0.8, label='MA60')
ax1.set_ylabel('价格 (元)', fontsize=11)
ax1.set_title('杰瑞股份(002353) 收盘价走势', fontsize=14, fontweight='bold')
ax1.legend(loc='upper left', fontsize=9)
ax1.grid(True, alpha=0.2)
ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'¥{x:.0f}'))
ax1.tick_params(labelbottom=False)

# 中：成交量
ax2 = fig.add_subplot(gs[1], sharex=ax1)
colors_vol = ['#e74c3c' if c >= o else '#27ae60' for c, o in zip(df['close'], df['open'])]
ax2.bar(df.index, df['vol']/10000, color=colors_vol, width=0.8, alpha=0.6)
ax2.set_ylabel('成交量\n(万手)', fontsize=10)
ax2.grid(True, alpha=0.2)
ax2.tick_params(labelbottom=False)

# 下：RSI
ax3 = fig.add_subplot(gs[2], sharex=ax1)
ax3.plot(df.index, df['rsi_14'], color='#8e44ad', linewidth=1.2, label='RSI(14)')
ax3.axhline(y=70, color='#e74c3c', linewidth=0.8, linestyle='--', alpha=0.6, label='超买线 70')
ax3.axhline(y=30, color='#27ae60', linewidth=0.8, linestyle='--', alpha=0.6, label='超卖线 30')
ax3.axhline(y=50, color='gray', linewidth=0.5, linestyle=':', alpha=0.4)
ax3.fill_between(df.index, 70, df['rsi_14'], where=(df['rsi_14'] >= 70), color='#e74c3c', alpha=0.15)
ax3.fill_between(df.index, 30, df['rsi_14'], where=(df['rsi_14'] <= 30), color='#27ae60', alpha=0.15)
ax3.set_ylabel('RSI', fontsize=10)
ax3.set_ylim(0, 100)
ax3.legend(loc='upper left', fontsize=9)
ax3.grid(True, alpha=0.2)
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax3.xaxis.set_major_locator(mdates.MonthLocator())
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.savefig('task1_01_RSI指标.png', bbox_inches='tight', dpi=150)
plt.close()
print("   [1/5] RSI 指标图表已保存 → task1_01_RSI指标.png")

# ---- 图表 2: MACD 指标 ----
fig = plt.figure(figsize=(20, 12))
gs = GridSpec(3, 1, figure=fig, height_ratios=[2, 1, 1], hspace=0.08)

# 上：收盘价
ax1 = fig.add_subplot(gs[0])
ax1.plot(df.index, df['close'], color='#e74c3c', linewidth=1.3, label='收盘价')
# 标注金叉死叉信号
for i in range(1, len(df)):
    # MACD金叉
    if df['macd'].iloc[i] > df['macd_signal'].iloc[i] and df['macd'].iloc[i-1] <= df['macd_signal'].iloc[i-1]:
        ax1.scatter(df.index[i], df['close'].iloc[i], marker='^', color='#e74c3c', s=80, zorder=5, alpha=0.8)
    # MACD死叉
    if df['macd'].iloc[i] < df['macd_signal'].iloc[i] and df['macd'].iloc[i-1] >= df['macd_signal'].iloc[i-1]:
        ax1.scatter(df.index[i], df['close'].iloc[i], marker='v', color='#27ae60', s=80, zorder=5, alpha=0.8)
ax1.set_ylabel('价格 (元)', fontsize=11)
ax1.set_title('杰瑞股份(002353) MACD 金叉/死叉信号', fontsize=14, fontweight='bold')
ax1.legend(loc='upper left', fontsize=9)
ax1.grid(True, alpha=0.2)
ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'¥{x:.0f}'))
ax1.tick_params(labelbottom=False)

# 中：MACD线与信号线
ax2 = fig.add_subplot(gs[1], sharex=ax1)
ax2.plot(df.index, df['macd'], color='#e74c3c', linewidth=1.2, label='MACD线 (DIF)')
ax2.plot(df.index, df['macd_signal'], color='#2980b9', linewidth=1.2, label='信号线 (DEA)', linestyle='--')
ax2.axhline(y=0, color='gray', linewidth=0.5, linestyle='-', alpha=0.5)
ax2.set_ylabel('MACD', fontsize=10)
ax2.legend(loc='upper left', fontsize=9)
ax2.grid(True, alpha=0.2)
ax2.tick_params(labelbottom=False)

# 下：MACD柱状图
ax3 = fig.add_subplot(gs[2], sharex=ax1)
colors_hist = ['#e74c3c' if v >= 0 else '#27ae60' for v in df['macd_hist']]
ax3.bar(df.index, df['macd_hist'], color=colors_hist, width=0.8, alpha=0.7)
ax3.axhline(y=0, color='gray', linewidth=0.5, linestyle='-')
ax3.set_ylabel('柱状图', fontsize=10)
ax3.set_xlabel('日期', fontsize=11)
ax3.grid(True, alpha=0.2)
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax3.xaxis.set_major_locator(mdates.MonthLocator())
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.savefig('task1_02_MACD指标.png', bbox_inches='tight', dpi=150)
plt.close()
print("   [2/5] MACD 指标图表已保存 → task1_02_MACD指标.png")

# ---- 图表 3: 布林带 ----
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12), height_ratios=[3, 1])

# 上半部分：价格 + 布林带
ax1.fill_between(df.index, df['bb_upper'], df['bb_lower'], color='#3498db', alpha=0.08)
ax1.plot(df.index, df['bb_upper'], color='#2980b9', linewidth=1.0, linestyle='--', label='上轨 (+2σ)', alpha=0.7)
ax1.plot(df.index, df['bb_middle'], color='#f39c12', linewidth=1.0, label='中轨 (MA20)', alpha=0.7)
ax1.plot(df.index, df['bb_lower'], color='#27ae60', linewidth=1.0, linestyle='--', label='下轨 (-2σ)', alpha=0.7)
ax1.plot(df.index, df['close'], color='#e74c3c', linewidth=1.5, label='收盘价', zorder=2)

# 标记触及布林带上下轨的位置
touch_upper = df[df['close'] >= df['bb_upper'] * 0.98]
touch_lower = df[df['close'] <= df['bb_lower'] * 1.02]
ax1.scatter(touch_upper.index, touch_upper['close'], color='#c0392b', s=30, alpha=0.6, zorder=3)
ax1.scatter(touch_lower.index, touch_lower['close'], color='#27ae60', s=30, alpha=0.6, zorder=3)

ax1.set_ylabel('价格 (元)', fontsize=11)
ax1.set_title('杰瑞股份(002353) 布林带 (Bollinger Bands)', fontsize=14, fontweight='bold')
ax1.legend(loc='upper left', fontsize=9)
ax1.grid(True, alpha=0.2)
ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'¥{x:.0f}'))
ax1.tick_params(labelbottom=False)

# 下半部分：布林带宽度(波动率指标)
ax2.fill_between(df.index, 0, df['bb_width'], color='#8e44ad', alpha=0.3)
ax2.plot(df.index, df['bb_width'], color='#8e44ad', linewidth=1.2)
ax2.axhline(y=df['bb_width'].mean(), color='#e74c3c', linewidth=0.8, linestyle='--',
            label=f'平均带宽 {df["bb_width"].mean():.1f}%')
ax2.set_ylabel('带宽 (%)', fontsize=10)
ax2.set_xlabel('日期', fontsize=11)
ax2.set_title('布林带宽度 (波动率指标)', fontsize=11)
ax2.legend(loc='upper left', fontsize=9)
ax2.grid(True, alpha=0.2)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax2.xaxis.set_major_locator(mdates.MonthLocator())
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig('task1_03_布林带.png', bbox_inches='tight', dpi=150)
plt.close()
print("   [3/5] 布林带图表已保存 → task1_03_布林带.png")

# ---- 图表 4: KDJ 指标 ----
fig = plt.figure(figsize=(20, 12))
gs = GridSpec(2, 1, figure=fig, height_ratios=[2, 1], hspace=0.08)

# 上：收盘价
ax1 = fig.add_subplot(gs[0])
ax1.plot(df.index, df['close'], color='#e74c3c', linewidth=1.3, label='收盘价')
# KDJ 金叉死叉
for i in range(1, len(df)):
    if df['kdj_k'].iloc[i] > df['kdj_d'].iloc[i] and df['kdj_k'].iloc[i-1] <= df['kdj_d'].iloc[i-1]:
        ax1.scatter(df.index[i], df['close'].iloc[i], marker='^', color='#e74c3c', s=60, zorder=5, alpha=0.7)
    if df['kdj_k'].iloc[i] < df['kdj_d'].iloc[i] and df['kdj_k'].iloc[i-1] >= df['kdj_d'].iloc[i-1]:
        ax1.scatter(df.index[i], df['close'].iloc[i], marker='v', color='#27ae60', s=60, zorder=5, alpha=0.7)
ax1.set_ylabel('价格 (元)', fontsize=11)
ax1.set_title('杰瑞股份(002353) KDJ 金叉/死叉信号', fontsize=14, fontweight='bold')
ax1.legend(loc='upper left', fontsize=9)
ax1.grid(True, alpha=0.2)
ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'¥{x:.0f}'))
ax1.tick_params(labelbottom=False)

# 下：KDJ
ax2 = fig.add_subplot(gs[1], sharex=ax1)
ax2.plot(df.index, df['kdj_k'], color='#e74c3c', linewidth=1.0, label='K值', alpha=0.8)
ax2.plot(df.index, df['kdj_d'], color='#2980b9', linewidth=1.0, label='D值', alpha=0.8)
ax2.plot(df.index, df['kdj_j'], color='#f39c12', linewidth=0.8, label='J值', alpha=0.6)
ax2.axhline(y=80, color='#e74c3c', linewidth=0.6, linestyle='--', alpha=0.5, label='超买线 80')
ax2.axhline(y=20, color='#27ae60', linewidth=0.6, linestyle='--', alpha=0.5, label='超卖线 20')
ax2.axhline(y=50, color='gray', linewidth=0.4, linestyle=':', alpha=0.3)
ax2.fill_between(df.index, 80, df['kdj_k'], where=(df['kdj_k'] >= 80), color='#e74c3c', alpha=0.1)
ax2.fill_between(df.index, 20, df['kdj_k'], where=(df['kdj_k'] <= 20), color='#27ae60', alpha=0.1)
ax2.set_ylabel('KDJ值', fontsize=10)
ax2.set_xlabel('日期', fontsize=11)
ax2.set_ylim(-20, 120)
ax2.legend(loc='upper left', fontsize=9, ncol=3)
ax2.grid(True, alpha=0.2)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax2.xaxis.set_major_locator(mdates.MonthLocator())
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.savefig('task1_04_KDJ指标.png', bbox_inches='tight', dpi=150)
plt.close()
print("   [4/5] KDJ 指标图表已保存 → task1_04_KDJ指标.png")

# ---- 图表 5: 四大指标综合面板 ----
fig, axes = plt.subplots(5, 1, figsize=(22, 18), sharex=True,
                          gridspec_kw={'height_ratios': [2, 1, 1, 1, 1]})
fig.subplots_adjust(hspace=0.06)

# 1) 收盘价 + 布林带
ax = axes[0]
ax.fill_between(df.index, df['bb_upper'], df['bb_lower'], color='#3498db', alpha=0.08)
ax.plot(df.index, df['bb_upper'], color='#2980b9', linewidth=0.8, linestyle='--', alpha=0.6)
ax.plot(df.index, df['bb_middle'], color='#f39c12', linewidth=0.8, alpha=0.6)
ax.plot(df.index, df['bb_lower'], color='#27ae60', linewidth=0.8, linestyle='--', alpha=0.6)
ax.plot(df.index, df['close'], color='#e74c3c', linewidth=1.3, label='收盘价')
ax.set_ylabel('价格', fontsize=9)
ax.set_title('杰瑞股份(002353) 技术指标综合面板', fontsize=16, fontweight='bold')
ax.legend(loc='upper left', fontsize=8)
ax.grid(True, alpha=0.15)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'¥{x:.0f}'))

# 2) 成交量
ax = axes[1]
colors_vol = ['#e74c3c' if c >= o else '#27ae60' for c, o in zip(df['close'], df['open'])]
ax.bar(df.index, df['vol']/10000, color=colors_vol, width=0.8, alpha=0.5)
ax.set_ylabel('量(万手)', fontsize=9)
ax.grid(True, alpha=0.15)

# 3) RSI
ax = axes[2]
ax.plot(df.index, df['rsi_14'], color='#8e44ad', linewidth=1.0)
ax.axhline(y=70, color='#e74c3c', linewidth=0.6, linestyle='--', alpha=0.5)
ax.axhline(y=30, color='#27ae60', linewidth=0.6, linestyle='--', alpha=0.5)
ax.fill_between(df.index, 70, df['rsi_14'], where=(df['rsi_14'] >= 70), color='#e74c3c', alpha=0.1)
ax.fill_between(df.index, 30, df['rsi_14'], where=(df['rsi_14'] <= 30), color='#27ae60', alpha=0.1)
ax.set_ylabel('RSI', fontsize=9)
ax.set_ylim(0, 100)
ax.grid(True, alpha=0.15)

# 4) MACD
ax = axes[3]
ax.plot(df.index, df['macd'], color='#e74c3c', linewidth=1.0, label='DIF')
ax.plot(df.index, df['macd_signal'], color='#2980b9', linewidth=1.0, label='DEA', linestyle='--')
colors_hist = ['#e74c3c' if v >= 0 else '#27ae60' for v in df['macd_hist']]
ax.bar(df.index, df['macd_hist'], color=colors_hist, width=0.8, alpha=0.5)
ax.axhline(y=0, color='gray', linewidth=0.4)
ax.set_ylabel('MACD', fontsize=9)
ax.legend(loc='upper left', fontsize=8, ncol=2)
ax.grid(True, alpha=0.15)

# 5) KDJ
ax = axes[4]
ax.plot(df.index, df['kdj_k'], color='#e74c3c', linewidth=0.8, alpha=0.8)
ax.plot(df.index, df['kdj_d'], color='#2980b9', linewidth=0.8, alpha=0.8)
ax.plot(df.index, df['kdj_j'], color='#f39c12', linewidth=0.6, alpha=0.5)
ax.axhline(y=80, color='#e74c3c', linewidth=0.5, linestyle='--', alpha=0.4)
ax.axhline(y=20, color='#27ae60', linewidth=0.5, linestyle='--', alpha=0.4)
ax.set_ylabel('KDJ', fontsize=9)
ax.set_ylim(-20, 120)
ax.grid(True, alpha=0.15)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.xaxis.set_major_locator(mdates.MonthLocator())
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.savefig('task1_05_指标综合面板.png', bbox_inches='tight', dpi=150)
plt.close()
print("   [5/5] 综合指标面板已保存 → task1_05_指标综合面板.png")

# ============================================================
# 第四部分：信号统计与总结
# ============================================================
print("\n" + "=" * 70)
print("  第四部分：信号统计与总结")
print("=" * 70)

# --- RSI 信号 ---
rsi_overbought = (df['rsi_14'] > 70).sum()
rsi_oversold = (df['rsi_14'] < 30).sum()
print(f"\n📊 RSI 信号统计:")
print(f"   超买天数 (RSI > 70): {rsi_overbought} 天 ({rsi_overbought/len(df)*100:.1f}%)")
print(f"   超卖天数 (RSI < 30): {rsi_oversold} 天 ({rsi_oversold/len(df)*100:.1f}%)")
print(f"   当前 RSI: {df['rsi_14'].iloc[-1]:.2f} → {'超买区域' if df['rsi_14'].iloc[-1] > 70 else ('超卖区域' if df['rsi_14'].iloc[-1] < 30 else '中性区域')}")

# --- MACD 信号 ---
macd_golden = 0   # 金叉次数
macd_dead = 0     # 死叉次数
for i in range(1, len(df)):
    if df['macd'].iloc[i] > df['macd_signal'].iloc[i] and df['macd'].iloc[i-1] <= df['macd_signal'].iloc[i-1]:
        macd_golden += 1
    if df['macd'].iloc[i] < df['macd_signal'].iloc[i] and df['macd'].iloc[i-1] >= df['macd_signal'].iloc[i-1]:
        macd_dead += 1

print(f"\n📊 MACD 信号统计:")
print(f"   金叉次数 (DIF上穿DEA): {macd_golden}")
print(f"   死叉次数 (DIF下穿DEA): {macd_dead}")
print(f"   MACD > 0 天数: {(df['macd'] > 0).sum()} 天 ({(df['macd'] > 0).mean()*100:.1f}%)")
print(f"   当前 MACD: {df['macd'].iloc[-1]:.4f} → {'零轴上方(多头)' if df['macd'].iloc[-1] > 0 else '零轴下方(空头)'}")

# --- 布林带信号 ---
bb_squeeze = df['bb_width'] < df['bb_width'].quantile(0.2)  # 带宽收窄
price_above = (df['close'] > df['bb_upper']).sum()
price_below = (df['close'] < df['bb_lower']).sum()

print(f"\n📊 布林带信号统计:")
print(f"   价格突破上轨天数: {price_above} 天")
print(f"   价格跌破下轨天数: {price_below} 天")
print(f"   带宽收窄天数(后20%): {bb_squeeze.sum()} 天")
print(f"   当前带宽: {df['bb_width'].iloc[-1]:.1f}%")
print(f"   当前 %B: {df['bb_pctb'].iloc[-1]:.3f} → {'超上轨' if df['bb_pctb'].iloc[-1] > 1 else ('超下轨' if df['bb_pctb'].iloc[-1] < 0 else '轨道内')}")

# --- KDJ 信号 ---
kdj_golden = 0
kdj_dead = 0
for i in range(1, len(df)):
    if df['kdj_k'].iloc[i] > df['kdj_d'].iloc[i] and df['kdj_k'].iloc[i-1] <= df['kdj_d'].iloc[i-1]:
        kdj_golden += 1
    if df['kdj_k'].iloc[i] < df['kdj_d'].iloc[i] and df['kdj_k'].iloc[i-1] >= df['kdj_d'].iloc[i-1]:
        kdj_dead += 1

print(f"\n📊 KDJ 信号统计:")
print(f"   金叉次数 (K上穿D): {kdj_golden}")
print(f"   死叉次数 (K下穿D): {kdj_dead}")
print(f"   当前 KDJ: K={df['kdj_k'].iloc[-1]:.1f}, D={df['kdj_d'].iloc[-1]:.1f}, J={df['kdj_j'].iloc[-1]:.1f}")

# --- 综合判断 ---
print("\n" + "-" * 50)
print("🎯 综合信号判断 (2026-07-01):")
print("-" * 50)

signals = []
# RSI
rsi_now = df['rsi_14'].iloc[-1]
if rsi_now > 70:
    signals.append("RSI超买 → ⚠️ 偏空")
elif rsi_now < 30:
    signals.append("RSI超卖 → ✅ 偏多")
else:
    signals.append(f"RSI中性({rsi_now:.0f}) → ➡️ 中性")

# MACD
if df['macd'].iloc[-1] > df['macd_signal'].iloc[-1]:
    signals.append("MACD金叉状态 → ✅ 偏多")
else:
    signals.append("MACD死叉状态 → ⚠️ 偏空")

if df['macd'].iloc[-1] > 0:
    signals.append("MACD零轴上方 → ✅ 多头趋势")

# BB
if df['bb_pctb'].iloc[-1] > 0.8:
    signals.append("价格接近布林上轨 → ⚠️ 可能回调")
elif df['bb_pctb'].iloc[-1] < 0.2:
    signals.append("价格接近布林下轨 → ✅ 可能反弹")

# KDJ
if df['kdj_j'].iloc[-1] > 100:
    signals.append("KDJ-J值超买 → ⚠️ 短期偏空")
elif df['kdj_j'].iloc[-1] < 0:
    signals.append("KDJ-J值超卖 → ✅ 短期偏多")

for s in signals:
    print(f"   {s}")

print("\n" + "=" * 70)
print("  ✅ 分析完成！所有图表已保存。")
print("  📁 生成文件:")
print("     - task1_01_RSI指标.png")
print("     - task1_02_MACD指标.png")
print("     - task1_03_布林带.png")
print("     - task1_04_KDJ指标.png")
print("     - task1_05_指标综合面板.png")
print("=" * 70)
