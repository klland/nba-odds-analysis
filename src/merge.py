"""
合併比賽結果與賠率資料
"""

import pandas as pd
import os
from difflib import SequenceMatcher

RAW_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
os.makedirs(PROCESSED_DIR, exist_ok=True)

# 球隊名稱對照（Basketball-Reference → Oddsportal）
TEAM_MAP = {
    'Atlanta Hawks': 'Atlanta Hawks',
    'Boston Celtics': 'Boston Celtics',
    'Brooklyn Nets': 'Brooklyn Nets',
    'Charlotte Hornets': 'Charlotte Hornets',
    'Chicago Bulls': 'Chicago Bulls',
    'Cleveland Cavaliers': 'Cleveland Cavaliers',
    'Dallas Mavericks': 'Dallas Mavericks',
    'Denver Nuggets': 'Denver Nuggets',
    'Detroit Pistons': 'Detroit Pistons',
    'Golden State Warriors': 'Golden State Warriors',
    'Houston Rockets': 'Houston Rockets',
    'Indiana Pacers': 'Indiana Pacers',
    'LA Clippers': 'Los Angeles Clippers',
    'Los Angeles Lakers': 'Los Angeles Lakers',
    'Memphis Grizzlies': 'Memphis Grizzlies',
    'Miami Heat': 'Miami Heat',
    'Milwaukee Bucks': 'Milwaukee Bucks',
    'Minnesota Timberwolves': 'Minnesota Timberwolves',
    'New Orleans Pelicans': 'New Orleans Pelicans',
    'New York Knicks': 'New York Knicks',
    'Oklahoma City Thunder': 'Oklahoma City Thunder',
    'Orlando Magic': 'Orlando Magic',
    'Philadelphia 76ers': 'Philadelphia 76ers',
    'Phoenix Suns': 'Phoenix Suns',
    'Portland Trail Blazers': 'Portland Trail Blazers',
    'Sacramento Kings': 'Sacramento Kings',
    'San Antonio Spurs': 'San Antonio Spurs',
    'Toronto Raptors': 'Toronto Raptors',
    'Utah Jazz': 'Utah Jazz',
    'Washington Wizards': 'Washington Wizards',
}

def fuzzy_match(name, candidates, threshold=0.6):
    best, best_score = None, 0
    for c in candidates:
        score = SequenceMatcher(None, name.lower(), c.lower()).ratio()
        if score > best_score:
            best_score, best = score, c
    return best if best_score >= threshold else None


def merge():
    games = pd.read_csv(os.path.join(RAW_DIR, 'nba_games.csv'))
    odds  = pd.read_csv(os.path.join(RAW_DIR, 'nba_odds.csv'))

    print(f'比賽資料：{len(games)} 筆')
    print(f'賠率資料：{len(odds)} 筆')

    # 用模糊比對合併
    odds_teams = odds['home_team'].unique().tolist()
    games['home_team_mapped'] = games['home_team'].apply(
        lambda x: fuzzy_match(TEAM_MAP.get(x, x), odds_teams)
    )
    games['away_team_mapped'] = games['away_team'].apply(
        lambda x: fuzzy_match(TEAM_MAP.get(x, x), odds_teams)
    )

    merged = games.merge(
        odds,
        left_on=['home_team_mapped', 'away_team_mapped'],
        right_on=['home_team', 'away_team'],
        how='inner'
    )

    print(f'合併後：{len(merged)} 筆')

    cols = ['date', 'season', 'home_team_x', 'away_team_x',
            'home_pts', 'away_pts', 'home_win', 'point_diff',
            'home_odd', 'away_odd', 'home_implied_prob', 'away_implied_prob', 'overround']
    merged = merged[[c for c in cols if c in merged.columns]]
    merged.columns = [c.replace('_x', '') for c in merged.columns]

    out = os.path.join(PROCESSED_DIR, 'merged.csv')
    merged.to_csv(out, index=False, encoding='utf-8-sig')
    print(f'✅ 儲存：{out}')
    return merged


if __name__ == '__main__':
    merge()
