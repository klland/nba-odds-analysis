"""
市場效率分析：運彩賠率 vs FiveThirtyEight Elo vs 實際比賽結果
分析：
  1. 賠率校準圖（市場隱含勝率 vs 實際勝率）
  2. 莊家 overround（抽水率）分布
  3. Elo vs 市場差異：誰更準？
  4. 背靠背賽程：市場是否充分定價？
  5. 賠率 vs Elo 分歧時的冷門率
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')
matplotlib.rcParams['font.family'] = 'Microsoft JhengHei'
matplotlib.rcParams['axes.unicode_minus'] = False

DATA   = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'odds_features.csv')
OUTPUT = os.path.join(os.path.dirname(__file__), '..', 'output')
os.makedirs(OUTPUT, exist_ok=True)


def load():
    df = pd.read_csv(DATA, parse_dates=['date'])
    df = df.dropna(subset=['home_fair_prob', 'home_win_prob', 'home_win'])
    return df


# ── 1. 校準圖：賠率 vs Elo vs 完美校準 ────────────────────
def plot_calibration(df):
    bins = np.arange(0.30, 0.85, 0.05)

    def get_calibration(prob_col):
        df['_bin'] = pd.cut(df[prob_col], bins=bins)
        g = df.groupby('_bin')['home_win'].agg(['mean', 'count']).reset_index()
        g['center'] = [iv.mid for iv in g['_bin']]
        return g

    g_mkt = get_calibration('home_fair_prob')
    g_elo = get_calibration('home_win_prob')

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.plot([0.3, 0.8], [0.3, 0.8], 'k--', alpha=0.4, label='完美校準線', linewidth=1.5)
    ax.scatter(g_mkt['center'], g_mkt['mean'],
               s=g_mkt['count'] * 0.5, color='crimson', alpha=0.8,
               label='賠率隱含勝率（去vig）', zorder=3)
    ax.plot(g_mkt['center'], g_mkt['mean'], '-', color='crimson', alpha=0.5)
    ax.scatter(g_elo['center'], g_elo['mean'],
               s=g_elo['count'] * 0.5, color='steelblue', alpha=0.8,
               label='FiveThirtyEight Elo 預測', zorder=3)
    ax.plot(g_elo['center'], g_elo['mean'], '-', color='steelblue', alpha=0.5)

    ax.set_xlabel('預測勝率', fontsize=13)
    ax.set_ylabel('實際勝率', fontsize=13)
    ax.set_title('校準比較：運彩賠率 vs Elo 模型\n（點大小 = 樣本數，2008–2015）', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT, '08_calibration_odds_vs_elo.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('✅ 圖表 08：校準比較')

    # 計算 Brier Score（越低越好）
    brier_mkt = ((df['home_fair_prob'] - df['home_win']) ** 2).mean()
    brier_elo = ((df['home_win_prob'] - df['home_win']) ** 2).mean()
    print(f'   Brier Score：賠率 {brier_mkt:.4f}  |  Elo {brier_elo:.4f}  （較小 = 更準確）')
    return brier_mkt, brier_elo


# ── 2. 莊家 Overround 分析 ─────────────────────────────────
def plot_overround(df):
    vig_pct = (df['overround'] - 1) * 100  # 轉成百分比

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # 分布直方圖
    axes[0].hist(vig_pct.dropna(), bins=40, color='steelblue', alpha=0.8, edgecolor='white')
    axes[0].axvline(vig_pct.mean(), color='crimson', linestyle='--', linewidth=2,
                    label=f'平均 {vig_pct.mean():.2f}%')
    axes[0].set_xlabel('Overround（%）', fontsize=12)
    axes[0].set_ylabel('場數', fontsize=12)
    axes[0].set_title('莊家抽水率分布', fontsize=14)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # 逐賽季趨勢
    season_vig = df.groupby('season').apply(
        lambda s: (s['overround'].mean() - 1) * 100
    ).reset_index(name='avg_vig')

    axes[1].plot(season_vig['season'], season_vig['avg_vig'], 'o-',
                 color='steelblue', linewidth=2, markersize=8)
    axes[1].axhline(season_vig['avg_vig'].mean(), color='gray', linestyle='--', alpha=0.5,
                    label=f'總平均 {season_vig["avg_vig"].mean():.2f}%')
    axes[1].set_xlabel('賽季', fontsize=12)
    axes[1].set_ylabel('平均 Overround（%）', fontsize=12)
    axes[1].set_title('莊家抽水率逐賽季趨勢', fontsize=14)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle('莊家抽水率（Overround）分析', fontsize=15, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT, '09_overround_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'✅ 圖表 09：Overround 分析  (平均 {vig_pct.mean():.2f}%，即每場下注虧損期望值約 {vig_pct.mean()/2:.2f}%)')


# ── 3. Elo 與市場分歧 vs 比賽結果 ─────────────────────────
def plot_divergence(df):
    """
    當 Elo 看好某隊（home_win_prob 高）但賠率不這麼認為（home_fair_prob 低），
    Elo 正確的機率是多少？反之亦然。
    """
    df = df.copy()
    # 把 prob_diff 分成 5 個區間
    df['diff_bin'] = pd.cut(df['prob_diff'], bins=[-0.5, -0.15, -0.05, 0.05, 0.15, 0.5])
    g = df.groupby('diff_bin').agg(
        actual_win_rate=('home_win', 'mean'),
        mkt_prob=('home_fair_prob', 'mean'),
        elo_prob=('home_win_prob', 'mean'),
        count=('home_win', 'count'),
    ).reset_index()
    g['bin_center'] = [iv.mid for iv in g['diff_bin']]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(len(g)), g['actual_win_rate'], color='steelblue', alpha=0.8, label='實際主場勝率')
    ax.plot(range(len(g)), g['mkt_prob'], 'r^--', markersize=8, label='賠率隱含勝率')
    ax.plot(range(len(g)), g['elo_prob'], 'g^--', markersize=8, label='Elo 勝率')
    ax.set_xticks(range(len(g)))
    ax.set_xticklabels([
        'Elo << 市場\n(Elo -15%+)',
        'Elo < 市場\n(Elo -5~15%)',
        '接近一致\n(±5%)',
        'Elo > 市場\n(Elo +5~15%)',
        'Elo >> 市場\n(Elo +15%+)',
    ], fontsize=10)
    for i, row in g.iterrows():
        ax.text(i, row['actual_win_rate'] + 0.015, f'{row["actual_win_rate"]:.1%}\n(n={row["count"]})',
                ha='center', fontsize=9)
    ax.set_ylabel('主場勝率', fontsize=12)
    ax.set_title('Elo 與賠率分歧程度 vs 實際比賽結果', fontsize=14)
    ax.legend(fontsize=11)
    ax.set_ylim(0.3, 0.85)
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT, '10_elo_vs_market_divergence.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('✅ 圖表 10：Elo vs 市場分歧分析')


# ── 4. 背靠背 × 賠率定價 ──────────────────────────────────
def plot_b2b_market_pricing(df):
    """市場是否已充分反映背靠背效應？"""
    df = df.copy()

    groups = {
        '雙方正常':    df[(df['home_b2b'] == 0) & (df['away_b2b'] == 0)],
        '主場背靠背':  df[(df['home_b2b'] == 1) & (df['away_b2b'] == 0)],
        '客場背靠背':  df[(df['home_b2b'] == 0) & (df['away_b2b'] == 1)],
        '雙方背靠背':  df[(df['home_b2b'] == 1) & (df['away_b2b'] == 1)],
    }

    labels   = list(groups.keys())
    actual   = [g['home_win'].mean()       for g in groups.values()]
    mkt_prob = [g['home_fair_prob'].mean() for g in groups.values()]
    elo_prob = [g['home_win_prob'].mean()  for g in groups.values()]
    counts   = [len(g) for g in groups.values()]

    x = np.arange(len(labels))
    width = 0.28

    fig, ax = plt.subplots(figsize=(12, 6))
    b1 = ax.bar(x - width, actual,   width, label='實際主場勝率', color='steelblue', alpha=0.85)
    b2 = ax.bar(x,         mkt_prob, width, label='賠率隱含勝率', color='crimson',   alpha=0.85)
    b3 = ax.bar(x + width, elo_prob, width, label='Elo 預測勝率', color='seagreen',  alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel('主場勝率', fontsize=12)
    ax.set_title('背靠背賽程：實際勝率 vs 賠率定價 vs Elo 預測', fontsize=14)
    ax.legend(fontsize=11)
    ax.set_ylim(0.45, 0.80)
    ax.grid(True, alpha=0.3, axis='y')

    for bars in [b1, b2, b3]:
        for i, bar in enumerate(bars):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f'{bar.get_height():.1%}', ha='center', fontsize=9)

    # 樣本數標注
    for i, n in enumerate(counts):
        ax.text(x[i], 0.465, f'n={n}', ha='center', fontsize=8.5, color='gray')

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT, '11_b2b_market_pricing.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('✅ 圖表 11：背靠背賽程 × 賠率定價')

    # 量化差距
    b2b_away = groups['客場背靠背']
    mkt_gap  = b2b_away['home_win'].mean() - b2b_away['home_fair_prob'].mean()
    print(f'   客場背靠背時：實際勝率 {b2b_away["home_win"].mean():.1%}，賠率隱含 {b2b_away["home_fair_prob"].mean():.1%}，差距 {mkt_gap:+.1%}')


# ── 5. 大冷門定價精準度 ───────────────────────────────────
def plot_upset_analysis(df):
    """賠率給的各勝率區間，實際表現如何？聚焦於冷門（<40%）"""
    bins = np.arange(0.05, 1.00, 0.10)
    df = df.copy()
    df['prob_bin'] = pd.cut(df['home_fair_prob'], bins=bins)
    g = df.groupby('prob_bin')['home_win'].agg(['mean', 'count']).reset_index()
    g['center'] = [iv.mid for iv in g['prob_bin']]
    g = g.dropna()

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['crimson' if c < 0.5 else 'steelblue' for c in g['center']]
    bars = ax.bar(range(len(g)), g['mean'], color=colors, alpha=0.8)
    ax.plot(range(len(g)), g['center'], 'k^--', markersize=7, label='賠率隱含勝率')
    ax.plot([0, 0], [0, 0], 's', color='crimson', label='主場為冷門（隱含<50%）')
    ax.plot([0, 0], [0, 0], 's', color='steelblue', label='主場為熱門（隱含>50%）')
    ax.set_xticks(range(len(g)))
    ax.set_xticklabels([f'{c:.0%}' for c in g['center']], fontsize=10)
    ax.set_xlabel('賠率隱含主場勝率區間', fontsize=12)
    ax.set_ylabel('實際主場勝率', fontsize=12)
    ax.set_title('冷門定價精準度：賠率隱含勝率 vs 實際勝率（2008–2015）', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    for i, row in g.iterrows():
        ax.text(list(range(len(g)))[list(g.index).index(i)],
                row['mean'] + 0.02,
                f'{row["mean"]:.0%}\nn={row["count"]}', ha='center', fontsize=8.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT, '12_upset_pricing.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('✅ 圖表 12：冷門定價精準度')


# ── 主程式 ────────────────────────────────────────────────
if __name__ == '__main__':
    print('載入合併資料...')
    df = load()
    print(f'  共 {len(df)} 場，賽季 {df["season"].min()}–{df["season"].max()}\n')

    print('=== 1. 校準比較（Brier Score）===')
    brier_mkt, brier_elo = plot_calibration(df)

    print('\n=== 2. 莊家 Overround ===')
    plot_overround(df)

    print('\n=== 3. Elo vs 市場分歧 ===')
    plot_divergence(df)

    print('\n=== 4. 背靠背 × 賠率定價 ===')
    plot_b2b_market_pricing(df)

    print('\n=== 5. 冷門定價精準度 ===')
    plot_upset_analysis(df)

    print('\n=== 總結 ===')
    print(f'Brier Score：賠率 {brier_mkt:.4f}  |  Elo {brier_elo:.4f}')
    if brier_mkt < brier_elo:
        print('→ 賠率校準性優於 Elo，市場效率較高')
    else:
        print('→ Elo 校準性優於賠率，Elo 包含市場未充分定價的資訊')
    print('\n🎉 市場效率分析完成！圖表在 output/')
