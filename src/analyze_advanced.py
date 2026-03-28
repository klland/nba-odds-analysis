"""
進階分析：完整特徵模型 + 逐賽季分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import os, sys

sys.stdout.reconfigure(encoding='utf-8')
matplotlib.rcParams['font.family'] = 'Microsoft JhengHei'
matplotlib.rcParams['axes.unicode_minus'] = False

DATA    = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'features.csv')
OUTPUT  = os.path.join(os.path.dirname(__file__), '..', 'output')
os.makedirs(OUTPUT, exist_ok=True)


def load():
    df = pd.read_csv(DATA)
    df['date'] = pd.to_datetime(df['date'])
    return df


# ── 1. 特徵重要性比較 ────────────────────────────────
def plot_feature_importance(df):
    features = ['home_win_prob', 'elo_diff', 'log_odds', 'home_b2b', 'away_b2b', 'b2b_advantage', 'month']
    df_clean = df.dropna(subset=features + ['home_win'])

    X = df_clean[features]
    y = df_clean['home_win']

    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X, y)

    importance = pd.Series(rf.feature_importances_, index=features).sort_values()
    feat_labels = {
        'home_win_prob': 'Elo 預測勝率',
        'elo_diff':      'Elo 差值',
        'log_odds':      'Log Odds',
        'home_b2b':      '主場背靠背',
        'away_b2b':      '客場背靠背',
        'b2b_advantage': '背靠背優勢差',
        'month':         '賽季月份',
    }
    importance.index = [feat_labels.get(i, i) for i in importance.index]

    fig, ax = plt.subplots(figsize=(9, 5))
    importance.plot(kind='barh', ax=ax, color='steelblue')
    ax.set_title('特徵重要性（隨機森林）', fontsize=15)
    ax.set_xlabel('重要性分數', fontsize=12)
    ax.grid(True, axis='x', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT, '04_feature_importance.png'), dpi=150, bbox_inches='tight')
    print('✅ 圖表 04：特徵重要性')
    plt.close()


# ── 2. 模型比較（基礎 vs 進階特徵）────────────────────
def compare_models(df):
    base_features  = ['home_win_prob']
    adv_features   = ['home_win_prob', 'elo_diff', 'home_b2b', 'away_b2b', 'b2b_advantage', 'month']

    df_clean = df.dropna(subset=adv_features + ['home_win'])
    y = df_clean['home_win']
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = {}
    configs = [
        ('基準線\n（永遠猜主場）', None, None),
        ('Elo 勝率\n（單一特徵）', LogisticRegression(), base_features),
        ('Elo + 背靠背\n（進階特徵）', LogisticRegression(), adv_features),
        ('隨機森林\n（進階特徵）', RandomForestClassifier(100, random_state=42), adv_features),
        ('梯度提升\n（進階特徵）', GradientBoostingClassifier(random_state=42), adv_features),
    ]

    accs = []
    for name, model, feats in configs:
        if model is None:
            accs.append(y.mean())
        else:
            X = df_clean[feats]
            scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
            accs.append(scores.mean())
        print(f'{name.replace(chr(10), " ")}: {accs[-1]:.4f}')
        results[name] = accs[-1]

    colors = ['gray', 'steelblue', 'royalblue', 'seagreen', 'darkorange']
    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.bar(results.keys(), results.values(), color=colors)
    ax.set_ylabel('5-Fold 交叉驗證準確率', fontsize=12)
    ax.set_title('各模型預測準確率比較', fontsize=15)
    ax.set_ylim(0.55, 0.75)
    for bar, val in zip(bars, results.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f'{val:.1%}', ha='center', fontsize=11, fontweight='bold')
    ax.axhline(y=y.mean(), color='gray', linestyle='--', alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT, '05_model_comparison_advanced.png'), dpi=150, bbox_inches='tight')
    print('✅ 圖表 05：進階模型比較')
    plt.close()
    return results


# ── 3. 逐賽季分析 ────────────────────────────────────
def plot_by_season(df):
    seasons = sorted(df['season'].unique())
    season_stats = []

    for s in seasons:
        sdf = df[df['season'] == s]
        home_wr   = sdf['home_win'].mean()
        elo_acc   = ((sdf['home_win_prob'] > 0.5) == sdf['home_win']).mean()
        upset_rate = sdf[sdf['home_win_prob'] < 0.5]['home_win'].mean()
        season_stats.append({
            'season':     s,
            'home_wr':    home_wr,
            'elo_acc':    elo_acc,
            'upset_rate': upset_rate,
            'games':      len(sdf),
        })

    stats = pd.DataFrame(season_stats)

    fig, axes = plt.subplots(3, 1, figsize=(13, 12), sharex=True)

    # 主場勝率趨勢
    axes[0].plot(stats['season'], stats['home_wr'], 'o-', color='steelblue', linewidth=2)
    axes[0].axhline(stats['home_wr'].mean(), color='gray', linestyle='--', alpha=0.5, label=f"平均 {stats['home_wr'].mean():.1%}")
    axes[0].set_ylabel('主場勝率', fontsize=12)
    axes[0].set_title('逐賽季分析（2005–2015 NBA 常規賽）', fontsize=15)
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0.5, 0.7)

    # Elo 預測準確率
    axes[1].plot(stats['season'], stats['elo_acc'], 'o-', color='seagreen', linewidth=2)
    axes[1].axhline(stats['elo_acc'].mean(), color='gray', linestyle='--', alpha=0.5, label=f"平均 {stats['elo_acc'].mean():.1%}")
    axes[1].set_ylabel('Elo 預測準確率', fontsize=12)
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0.58, 0.72)

    # 冷門發生率（Elo 預測客場贏但主場贏）
    axes[2].bar(stats['season'], stats['upset_rate'], color='salmon', alpha=0.8)
    axes[2].axhline(stats['upset_rate'].mean(), color='gray', linestyle='--', alpha=0.5, label=f"平均 {stats['upset_rate'].mean():.1%}")
    axes[2].set_ylabel('冷門率（預測客勝→實際主勝）', fontsize=12)
    axes[2].set_xlabel('賽季', fontsize=12)
    axes[2].legend(); axes[2].grid(True, alpha=0.3, axis='y')

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT, '06_season_trends.png'), dpi=150, bbox_inches='tight')
    print('✅ 圖表 06：逐賽季趨勢')
    plt.close()

    print('\n逐賽季摘要：')
    print(stats.to_string(index=False))
    return stats


# ── 4. 背靠背效應分析 ────────────────────────────────
def plot_b2b_effect(df):
    groups = {
        '雙方正常': df[(df['home_b2b'] == 0) & (df['away_b2b'] == 0)]['home_win'].mean(),
        '主場背靠背': df[(df['home_b2b'] == 1) & (df['away_b2b'] == 0)]['home_win'].mean(),
        '客場背靠背': df[(df['home_b2b'] == 0) & (df['away_b2b'] == 1)]['home_win'].mean(),
        '雙方背靠背': df[(df['home_b2b'] == 1) & (df['away_b2b'] == 1)]['home_win'].mean(),
    }

    counts = {
        '雙方正常': len(df[(df['home_b2b'] == 0) & (df['away_b2b'] == 0)]),
        '主場背靠背': len(df[(df['home_b2b'] == 1) & (df['away_b2b'] == 0)]),
        '客場背靠背': len(df[(df['home_b2b'] == 0) & (df['away_b2b'] == 1)]),
        '雙方背靠背': len(df[(df['home_b2b'] == 1) & (df['away_b2b'] == 1)]),
    }

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ['steelblue', 'salmon', 'seagreen', 'gray']
    bars = ax.bar(groups.keys(), groups.values(), color=colors, alpha=0.85)
    ax.axhline(df['home_win'].mean(), color='black', linestyle='--', alpha=0.5, label=f"整體主場勝率 {df['home_win'].mean():.1%}")
    ax.set_ylabel('主場勝率', fontsize=12)
    ax.set_title('背靠背賽程對主場勝率的影響', fontsize=15)
    ax.set_ylim(0.45, 0.75)
    ax.legend()
    for bar, (label, val) in zip(bars, groups.items()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{val:.1%}\n(n={counts[label]})', ha='center', fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT, '07_b2b_effect.png'), dpi=150, bbox_inches='tight')
    print('✅ 圖表 07：背靠背效應')
    plt.close()

    print('\n背靠背效應摘要：')
    for k, v in groups.items():
        print(f'  {k}: 主場勝率 {v:.1%} (n={counts[k]})')


# ── 主程式 ────────────────────────────────────────────
if __name__ == '__main__':
    print('載入資料...')
    df = load()
    print(f'共 {len(df)} 筆\n')

    print('=== 特徵重要性 ===')
    plot_feature_importance(df)

    print('\n=== 模型比較（5-Fold CV）===')
    compare_models(df)

    print('\n=== 逐賽季分析 ===')
    plot_by_season(df)

    print('\n=== 背靠背效應 ===')
    plot_b2b_effect(df)

    print('\n🎉 進階分析完成！圖表在 output/')
