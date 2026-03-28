"""
分析賠率準確性與建立預測模型
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import os

matplotlib.rcParams['font.family'] = 'Microsoft JhengHei'  # 支援中文
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data():
    base = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
    return pd.read_csv(os.path.join(base, 'merged.csv'))


def plot_odds_accuracy(df: pd.DataFrame):
    """分析 Elo 預測勝率 vs 實際勝率（校準圖）"""
    bins = np.arange(0.3, 0.85, 0.05)
    df['prob_bin'] = pd.cut(df['home_win_prob'], bins=bins)

    grouped = df.groupby('prob_bin')['home_win'].agg(['mean', 'count']).reset_index()
    grouped.columns = ['prob_bin', 'actual_win_rate', 'count']
    grouped['bin_center'] = [iv.mid for iv in grouped['prob_bin']]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(grouped['bin_center'], grouped['actual_win_rate'],
               s=grouped['count'] * 0.5, alpha=0.7, color='steelblue', label='實際勝率')
    ax.plot([0.3, 0.8], [0.3, 0.8], 'r--', label='完美校準線')
    ax.set_xlabel('Elo 預測勝率（類賠率隱含勝率）', fontsize=13)
    ax.set_ylabel('實際勝率', fontsize=13)
    ax.set_title('FiveThirtyEight Elo 模型校準分析\n（預測勝率 vs 實際勝率）', fontsize=15)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(OUTPUT_DIR, '01_elo_calibration.png'), dpi=150, bbox_inches='tight')
    print('✅ 圖表 01：Elo 校準分析')
    plt.close()


def plot_home_advantage(df: pd.DataFrame):
    """主場優勢分析"""
    home_win_rate = df['home_win'].mean()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 整體主場勝率
    axes[0].bar(['主場', '客場'], [home_win_rate, 1 - home_win_rate],
                color=['steelblue', 'salmon'])
    axes[0].set_title('整體主客場勝率', fontsize=13)
    axes[0].set_ylabel('勝率')
    for i, v in enumerate([home_win_rate, 1 - home_win_rate]):
        axes[0].text(i, v + 0.01, f'{v:.1%}', ha='center', fontsize=12)

    # Elo vs 主場優勢
    df['market_favorite'] = df['home_win_prob'] > 0.5
    upset_rate = df[~df['market_favorite']]['home_win'].mean()
    fav_rate = df[df['market_favorite']]['home_win'].mean()
    axes[1].bar(['市場看好主場', '市場看好客場（冷門）'],
                [fav_rate, upset_rate], color=['steelblue', 'orange'])
    axes[1].set_title('市場預測方向 vs 實際勝率', fontsize=13)
    axes[1].set_ylabel('主場勝率')
    for i, v in enumerate([fav_rate, upset_rate]):
        axes[1].text(i, v + 0.01, f'{v:.1%}', ha='center', fontsize=12)

    fig.suptitle('主場優勢分析', fontsize=15)
    fig.savefig(os.path.join(OUTPUT_DIR, '02_home_advantage.png'), dpi=150, bbox_inches='tight')
    print('✅ 圖表 02：主場優勢')
    plt.close()


def build_model(df: pd.DataFrame):
    """建立預測模型"""
    features = ['home_win_prob', 'home_elo', 'away_elo']
    df_clean = df.dropna(subset=features + ['home_win'])

    X = df_clean[features]
    y = df_clean['home_win']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        '邏輯迴歸': LogisticRegression(),
        '隨機森林': RandomForestClassifier(n_estimators=100, random_state=42),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc
        print(f'\n{name} 準確率：{acc:.4f}')
        print(classification_report(y_test, y_pred, target_names=['客場勝', '主場勝']))

    # 畫比較圖
    fig, ax = plt.subplots(figsize=(7, 5))
    baseline = y_test.mean()  # 永遠猜主場勝的基準
    bars = ax.bar(list(results.keys()) + ['基準線（永遠猜主場）'],
                  list(results.values()) + [baseline],
                  color=['steelblue', 'seagreen', 'gray'])
    ax.set_ylabel('準確率', fontsize=13)
    ax.set_title('預測模型比較', fontsize=15)
    ax.set_ylim(0.5, 0.75)
    for bar, val in zip(bars, list(results.values()) + [baseline]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f'{val:.1%}', ha='center', fontsize=11)
    fig.savefig(os.path.join(OUTPUT_DIR, '03_model_comparison.png'), dpi=150, bbox_inches='tight')
    print('\n✅ 圖表 03：模型比較')
    plt.close()

    return results


if __name__ == '__main__':
    print('載入資料...')
    df = load_data()
    print(f'共 {len(df)} 筆資料\n')

    print('=== 賠率校準分析 ===')
    plot_odds_accuracy(df)

    print('\n=== 主場優勢分析 ===')
    plot_home_advantage(df)

    print('\n=== 預測模型 ===')
    build_model(df)

    print('\n🎉 分析完成，圖表在 output/ 資料夾')
