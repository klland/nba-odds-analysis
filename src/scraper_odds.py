"""
爬取 Oddsportal NBA 歷史賠率
取得各場比賽的主客隊賠率（歐洲盤）
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
import time
import os

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xhtml+xml',
    'Accept-Language': 'en-US,en;q=0.9',
    'Referer': 'https://www.oddsportal.com/',
}

def odds_to_implied_prob(odd: float) -> float:
    """歐洲盤賠率轉隱含勝率"""
    if odd <= 1:
        return 0
    return round(1 / odd, 4)

def scrape_oddsportal_season(season_str: str) -> pd.DataFrame:
    """
    爬取 Oddsportal 賽季賠率
    season_str: 例如 '2023-2024'
    """
    url = f'https://www.oddsportal.com/basketball/usa/nba-{season_str}/results/'
    print(f'爬取賠率：{url}')

    all_odds = []
    page = 1

    while True:
        page_url = f'{url}#/page/{page}/' if page > 1 else url
        try:
            resp = requests.get(page_url, headers=HEADERS, timeout=15)
            if resp.status_code != 200:
                print(f'  頁面 {page} 失敗（{resp.status_code}）')
                break

            soup = BeautifulSoup(resp.text, 'lxml')

            # 找比賽列表
            rows = soup.select('div.eventRow')
            if not rows:
                rows = soup.select('tr.deactivate')

            if not rows:
                print(f'  頁面 {page} 無資料，停止')
                break

            for row in rows:
                try:
                    teams = row.select('.participant-name')
                    odds_els = row.select('.odds-nowrap span, .oddsCell span')

                    if len(teams) < 2 or len(odds_els) < 2:
                        continue

                    home = teams[0].text.strip()
                    away = teams[1].text.strip()

                    home_odd = float(odds_els[0].text.strip())
                    away_odd = float(odds_els[1].text.strip())

                    all_odds.append({
                        'home_team':       home,
                        'away_team':       away,
                        'home_odd':        home_odd,
                        'away_odd':        away_odd,
                        'home_implied_prob': odds_to_implied_prob(home_odd),
                        'away_implied_prob': odds_to_implied_prob(away_odd),
                        'overround':       round(odds_to_implied_prob(home_odd) + odds_to_implied_prob(away_odd), 4),
                    })
                except (ValueError, IndexError):
                    continue

            print(f'  頁面 {page}：{len(rows)} 筆')
            page += 1
            time.sleep(3)

        except requests.RequestException as e:
            print(f'  錯誤：{e}')
            break

    return pd.DataFrame(all_odds)


if __name__ == '__main__':
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
    os.makedirs(output_dir, exist_ok=True)

    seasons = ['2022-2023', '2023-2024']
    all_dfs = []

    for season in seasons:
        df = scrape_oddsportal_season(season)
        if not df.empty:
            all_dfs.append(df)
            print(f'  {season}：{len(df)} 筆賠率')
        time.sleep(5)

    if all_dfs:
        result = pd.concat(all_dfs, ignore_index=True)
        path = os.path.join(output_dir, 'nba_odds.csv')
        result.to_csv(path, index=False, encoding='utf-8-sig')
        print(f'\n✅ 儲存：{path}（{len(result)} 筆）')
    else:
        print('❌ 沒有取得賠率資料')
