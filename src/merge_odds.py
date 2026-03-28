"""
合併 SBR 賠率資料與 FiveThirtyEight Elo 特徵
輸出：data/processed/odds_features.csv
"""

import pandas as pd
import numpy as np
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

ODDS_PATH  = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw',       'nba_odds_sbr.csv')
ELO_PATH   = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'features.csv')
OUTPUT     = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'odds_features.csv')

# SBR 隊名 → Elo 縮寫（歷史球隊名稱也納入）
TEAM_MAP = {
    'Atlanta':      'ATL', 'Boston':       'BOS', 'Brooklyn':     'BRK',
    'NewJersey':    'NJN', 'Charlotte':    'CHA', 'Chicago':      'CHI',
    'Cleveland':    'CLE', 'Dallas':       'DAL', 'Denver':       'DEN',
    'Detroit':      'DET', 'GoldenState':  'GSW', 'Houston':      'HOU',
    'Indiana':      'IND', 'LAClippers':   'LAC', 'LALakers':     'LAL',
    'Memphis':      'MEM', 'Miami':        'MIA', 'Milwaukee':    'MIL',
    'Minnesota':    'MIN', 'NewOrleans':   'NOP', 'NewYork':      'NYK',
    'OklahomaCity': 'OKC', 'Orlando':      'ORL', 'Philadelphia': 'PHI',
    'Phoenix':      'PHO', 'Portland':     'POR', 'Sacramento':   'SAC',
    'SanAntonio':   'SAS', 'Seattle':      'SEA', 'Toronto':      'TOR',
    'Utah':         'UTA', 'Washington':   'WAS',
    # 歷史球隊（New Orleans Hornets / Oklahoma City Thunder 前身）
    'NOKOklahoma':  'NOK', 'NOHornets':    'NOH',
}


def load_and_clean_odds() -> pd.DataFrame:
    df = pd.read_csv(ODDS_PATH, parse_dates=['date'])
    df['home_abbr'] = df['home_team'].map(TEAM_MAP)
    df['away_abbr'] = df['away_team'].map(TEAM_MAP)

    # 未對應的隊名
    missing = df[df['home_abbr'].isna()]['home_team'].unique()
    if len(missing):
        print(f'  未對應主場隊名：{list(missing)}')

    df = df.dropna(subset=['home_abbr', 'away_abbr'])
    return df


def load_elo() -> pd.DataFrame:
    df = pd.read_csv(ELO_PATH, parse_dates=['date'])
    return df


def merge(odds: pd.DataFrame, elo: pd.DataFrame) -> pd.DataFrame:
    """
    匹配鍵：date + home_team + away_team（使用縮寫）
    """
    odds_key = odds[['date', 'season', 'home_abbr', 'away_abbr',
                      'home_ml_prob', 'away_ml_prob', 'home_fair_prob',
                      'overround', 'open_spread', 'close_spread',
                      'open_total', 'close_total']].copy()
    odds_key.rename(columns={'home_abbr': 'home_team', 'away_abbr': 'away_team'}, inplace=True)

    merged = pd.merge(
        elo, odds_key,
        on=['date', 'home_team', 'away_team'],
        how='inner',
        suffixes=('', '_odds'),
    )

    # 衍生欄位
    # 1. 賠率隱含勝率 vs Elo 勝率差（正數 = 賠率更看好主場）
    merged['prob_diff'] = (merged['home_fair_prob'] - merged['home_win_prob']).round(4)

    # 2. 「市場低估」指標：Elo 勝率高但賠率隱含勝率低
    merged['elo_over_market'] = (merged['home_win_prob'] > merged['home_fair_prob']).astype(int)

    # 3. 讓分轉換的隱含勝率（高斯分布近似：spread/sqrt(2)*sigma）
    #    NBA 每分約對應 2.5% 勝率，粗估：close_spread/14 + 0.5
    merged['spread_implied_prob'] = (0.5 - merged['close_spread'] / 14).clip(0.05, 0.95).round(4)

    return merged


if __name__ == '__main__':
    print('載入賠率資料...')
    odds = load_and_clean_odds()
    print(f'  賠率：{len(odds)} 場比賽')

    print('載入 Elo 特徵資料...')
    elo = load_elo()
    print(f'  Elo：{len(elo)} 場比賽')

    print('合併中...')
    merged = merge(odds, elo)
    print(f'  合併後：{len(merged)} 場（匹配率 {len(merged)/len(odds):.1%}）')

    merged.to_csv(OUTPUT, index=False, encoding='utf-8-sig')
    print(f'✅ 儲存：{OUTPUT}')
    print(f'   欄位數：{merged.shape[1]}')
    print(f'   賽季範圍：{merged["season"].min()} – {merged["season"].max()}')

    # 基本統計
    print('\n=== 市場隱含勝率基本統計 ===')
    print(f'賠率公平勝率平均：{merged["home_fair_prob"].mean():.3f}')
    print(f'Elo 預測勝率平均：{merged["home_win_prob"].mean():.3f}')
    print(f'Elo 與賠率差值（絕對值）平均：{merged["prob_diff"].abs().mean():.4f}')
    print(f'平均 overround（莊家抽水）：{(merged["overround"] - 1).mean():.2%}')
