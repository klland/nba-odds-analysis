"""
特徵工程：加入背靠背賽程、近期勝率、Elo 差值等特徵
"""

import pandas as pd
import numpy as np
import sys, os

sys.stdout.reconfigure(encoding='utf-8')

INPUT  = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'merged.csv')
OUTPUT = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'features.csv')


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # ── 1. Elo 差值（主場強弱相對指標）
    df['elo_diff'] = df['home_elo'] - df['away_elo']

    # ── 2. 預測值轉換：log odds
    df['log_odds'] = np.log(df['home_win_prob'] / (1 - df['home_win_prob'] + 1e-9))

    # ── 3. 冷門場（Elo 預測客場贏）
    df['is_upset_game'] = (df['home_win_prob'] < 0.5).astype(int)

    # ── 4. 背靠背賽程（前一天也有比賽）
    all_dates = pd.concat([
        df[['date', 'home_team']].rename(columns={'home_team': 'team'}),
        df[['date', 'away_team']].rename(columns={'away_team': 'team'}),
    ])

    def is_b2b(team, date):
        prev = date - pd.Timedelta(days=1)
        return int(((all_dates['team'] == team) & (all_dates['date'] == prev)).any())

    print('計算背靠背賽程...')
    df['home_b2b'] = [is_b2b(row.home_team, row.date) for row in df.itertuples()]
    df['away_b2b'] = [is_b2b(row.away_team, row.date) for row in df.itertuples()]
    df['b2b_advantage'] = df['away_b2b'] - df['home_b2b']  # 正數 = 主場有背靠背優勢

    # ── 5. 近 10 場勝率（滾動）
    print('計算近期勝率...')

    def rolling_win_rate(df, team_col, result_col, window=10):
        rates = {}
        for team in df[team_col].unique():
            mask = df[team_col] == team
            team_df = df[mask].copy()
            rates[team] = team_df[result_col].shift(1).rolling(window, min_periods=3).mean()
            df.loc[mask, f'{team_col}_recent_wr'] = rates[team].values
        return df

    # 主場隊近期主場勝率
    df['home_win_float'] = df['home_win'].astype(float)
    home_recent = {}
    away_recent = {}
    for team in df['home_team'].unique():
        mask = df['home_team'] == team
        home_recent[team] = df.loc[mask, 'home_win_float'].shift(1).rolling(10, min_periods=3).mean()

    df['home_recent_wr'] = pd.concat(
        [home_recent[t] for t in df['home_team'].unique()],
        axis=0
    ).sort_index()

    # ── 6. 得分差（後驗，僅供分析用，不用於預測）
    df['point_diff'] = df['home_pts'] - df['away_pts']

    # ── 7. 賽季進度（月份）
    df['month'] = df['date'].dt.month

    print(f'特徵工程完成，共 {len(df)} 筆，{df.shape[1]} 個欄位')
    print('新增欄位：', ['elo_diff', 'log_odds', 'is_upset_game', 'home_b2b', 'away_b2b', 'b2b_advantage', 'home_recent_wr', 'month'])

    df.to_csv(OUTPUT, index=False, encoding='utf-8-sig')
    print(f'✅ 儲存：{OUTPUT}')
    return df


if __name__ == '__main__':
    df = pd.read_csv(INPUT)
    add_features(df)
