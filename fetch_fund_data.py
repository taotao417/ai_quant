"""fetch_fund_data.py —— 把 Tushare MCP 返回的 JSON 结果文件，
规范化为与 杰瑞股份_日线_前复权.csv 同构的 CSV 数据储备。

输出目录：/Users/mac/Desktop/量化实践/fund_data/
统一列：date,open,high,low,close,pre_close,change,pct_chg,vol,amount
- ETF / LOF：直接取 fund_daily 的 OHLC（市场价）。
- 场外基金：close = 复权净值(adj_nav，含分红再投资，总收益口径)；
            open=high=low=close=adj_nav（场外基金无盘中高低，单日净值即成交价）；
            vol/amount 留空。next-day open == T+1 净值，对应场外基金 T 日收盘后
            下单、T+1 净值成交的实际情况。

定位策略：按 ts_code 内容匹配 tool-results 下所有 fund_nav-* / fund_daily-* 文件，
取 mtime 最新的一份（重新抓取后自动用新数据，不依赖固定文件名前缀）。
"""
from __future__ import annotations
import json
import os
import glob

BASE = "/Users/mac/.workbuddy/projects/Users-mac-Desktop-量化实践/054fb848-7ec6-479c-a401-a583f04b8a77/tool-results"
OUTDIR = "/Users/mac/Desktop/量化实践/fund_data"
os.makedirs(OUTDIR, exist_ok=True)


def _find_latest(target_code: str, kind: str) -> str:
    pat = os.path.join(BASE, f"mcp-connector-proxy-tushareMcp_fund_{kind}-*")
    cands = glob.glob(pat)
    best = None
    for p in cands:
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        if not isinstance(data, list):
            data = data.get("data", data.get("items", []))
        if not data:
            continue
        rec0 = data[0]
        if isinstance(rec0, dict) and rec0.get("ts_code") == target_code:
            mtime = os.path.getmtime(p)
            if best is None or mtime > best[1]:
                best = (p, mtime)
    if best is None:
        raise FileNotFoundError(target_code)
    return best[0]


# (ts_code, 输出文件名, 类型 nav/daily, 名称)
JOBS = [
    ("022430.OF", "022430_中证A500.csv", "nav", "中证A500"),
    ("017953.OF", "017953_中证1000指数增强.csv", "nav", "中证1000指数增强"),
    ("012970.OF", "012970_半导体芯片.csv", "nav", "半导体芯片"),
    ("004432.OF", "004432_有色金属.csv", "nav", "有色金属"),
    ("021958.OF", "021958_黄金产业股票.csv", "nav", "黄金产业股票"),
    ("018392.OF", "018392_黄金基金.csv", "nav", "黄金基金"),
    ("004320.OF", "004320_乐享生活.csv", "nav", "乐享生活"),
    ("161725.SZ", "161725_白酒基金LOF.csv", "daily", "白酒基金(LOF)"),
    ("159865.SZ", "159865_养殖ETF.csv", "daily", "养殖ETF"),
    ("512660.SH", "512660_军工ETF.csv", "daily", "军工ETF"),
    ("588200.SH", "588200_科创芯片ETF.csv", "daily", "科创芯片ETF"),
    ("513050.SH", "513050_中概互联ETF.csv", "daily", "中概互联ETF"),
]

manifest = []


def normalize_nav(rows):
    rows = sorted(rows, key=lambda r: str(r["nav_date"]))
    out = []
    prev = None
    for r in rows:
        d = str(r["nav_date"])
        adj = r.get("adj_nav")
        price = float(adj) if adj is not None else float(r["unit_nav"])
        pre = prev if prev is not None else price
        chg = price - pre
        pct = (chg / pre * 100.0) if pre else 0.0
        out.append({
            "date": d, "open": price, "high": price, "low": price, "close": price,
            "pre_close": round(pre, 6), "change": round(chg, 6),
            "pct_chg": round(pct, 4), "vol": "", "amount": "",
        })
        prev = price
    return out


def normalize_daily(rows):
    rows = sorted(rows, key=lambda r: str(r["trade_date"]))
    out = []
    for r in rows:
        out.append({
            "date": str(r["trade_date"]),
            "open": float(r["open"]), "high": float(r["high"]), "low": float(r["low"]),
            "close": float(r["close"]), "pre_close": float(r["pre_close"]),
            "change": float(r["change"]), "pct_chg": float(r["pct_chg"]),
            "vol": float(r["vol"]), "amount": float(r["amount"]),
        })
    return out


HEADER = ["date", "open", "high", "low", "close", "pre_close", "change", "pct_chg", "vol", "amount"]

for code, out_name, kind, name in JOBS:
    path = _find_latest(code, kind)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        data = data.get("data", data.get("items", []))
    if kind == "nav":
        out = normalize_nav(data)
    else:
        out = normalize_daily(data)
    out_path = os.path.join(OUTDIR, out_name)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        f.write(",".join(HEADER) + "\n")
        for row in out:
            f.write(",".join(str(row[c]) for c in HEADER) + "\n")
    first, last = out[0], out[-1]
    tr = (float(last["close"]) / float(first["close"]) - 1.0) * 100.0
    manifest.append({
        "ts_code": code, "file": out_name, "name": name, "kind": kind, "rows": len(out),
        "start": first["date"], "end": last["date"],
        "first_close": round(float(first["close"]), 4),
        "last_close": round(float(last["close"]), 4),
        "total_return_pct": round(tr, 2),
    })
    print(f"{out_name:28s} {kind:5s} rows={len(out):4d}  {first['date']}..{last['date']}  return={tr:7.2f}%")

with open(os.path.join(OUTDIR, "manifest.json"), "w", encoding="utf-8") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2)
print("\nmanifest.json written:", os.path.join(OUTDIR, "manifest.json"))
