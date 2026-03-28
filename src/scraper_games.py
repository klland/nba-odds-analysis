"""
爬取 Basketball-Reference NBA 比賽結果
目標：取得每場比賽的主客隊、得分、勝負
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
}

def scrape_season(season_year: int) -> pd.DataFrame:
    """
    爬取指定賽季的所有比賽結果
    season_year: 賽季結束年份，例如 2024 代表 2023-24 賽季
    """
    months = ['october', 'november', 'december', 'january', 'february', 'march', 'april']
    all_games = []

    for month in months:
        url = f'https://www.basketball-reference.com/leagues/NBA_{season_year}_games-{month}.html'
        print(f'  爬取 {season_year} 賽季 {month}...')

        try:
            resp = requests.get(url, headers=HEADERS, timeout=10)
            if resp.status_code != 200:
                print(f'    跳過（{resp.status_code}）')
                continue

            soup = BeautifulSoup(resp.text, 'lxml')
            table = soup.find('table', id='schedule')
            if not table:
                continue

            for row in table.find('tbody').find_all('tr'):
                if row.get('class') == ['thead']:
                    continue
                tds = row.find_all('td')
                if len(tds) < 5:
                    continue

                try:
                    date      = row.find('th').text.strip()
                    away_team = tds[1].text.strip()  # Los Angeles Lakers
                    away_pts  = tds[2].text.strip()  # 107
                    home_team = tds[3].text.strip()  # Denver Nuggets
                    home_pts  = tds[4].text.strip()  # 119

                    if not away_pts or not home_pts:
                        continue  # 未來場次

                    away_pts = int(away_pts)
                    home_pts = int(home_pts)

                    all_games.append({
                        'date':      date,
                        'season':    season_year,
                        'away_team': away_team,
                        'home_team': home_team,
                        'away_pts':  away_pts,
                        'home_pts':  home_pts,
                        'home_win':  1 if home_pts > away_pts else 0,
                        'point_diff': home_pts - away_pts,
                    })
                except (ValueError, AttributeError):
                    continue

            time.sleep(3)  # 避免被封鎖

        except requests.RequestException as e:
            print(f'    錯誤：{e}')
            continue

    return pd.DataFrame(all_games)


def scrape_multiple_seasons(start: int, end: int) -> pd.DataFrame:
    """爬取多個賽季"""
    all_dfs = []
    for year in range(start, end + 1):
        print(f'\n=== 賽季 {year-1}-{str(year)[2:]} ===')
        df = scrape_season(year)
        if not df.empty:
            all_dfs.append(df)
            print(f'  取得 {len(df)} 場比賽')
    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()


if __name__ == '__main__':
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
    os.makedirs(output_dir, exist_ok=True)

    # 爬取近 3 個賽季（2022-23、2023-24、2024-25）
    df = scrape_multiple_seasons(2023, 2025)

    if not df.empty:
        output_path = os.path.join(output_dir, 'nba_games.csv')
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f'\n✅ 儲存完成：{output_path}')
        print(f'   共 {len(df)} 場比賽')
        print(df.head())
    else:
        print('❌ 沒有取得任何資料')
