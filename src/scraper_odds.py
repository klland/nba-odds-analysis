"""
從 sportsbookreviewsonline.com 爬取 NBA 歷史賠率（HTML 表格）
欄位：Date, Rot, VH, Team, 1st-4th, Final, Open, Close, ML, 2H
每兩行為一場比賽：V(客隊) + H(主隊)
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import sys
import time

sys.stdout.reconfigure(encoding='utf-8')

OUTPUT = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'nba_odds_sbr.csv')
BASE_URL = 'https://www.sportsbookreviewsonline.com/scoresoddsarchives/nba-odds-{}'
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

# Elo dataset 的 season 欄位（賽季結束年份）→ SBR URL 後綴
SEASON_MAP = {
    2006: '2005-06',
    2007: '2006-07',
    2008: '2007-08',
    2009: '2008-09',
    2010: '2009-10',
    2011: '2010-11',
    2012: '2011-12',
    2013: '2012-13',
    2014: '2013-14',
    2015: '2014-15',
}


def ml_to_prob(ml_str: str) -> float:
    """Moneyline 字串轉隱含勝率（含 vig）"""
    try:
        ml = float(str(ml_str).replace('pk', '0').strip())
        if ml == 0:
            return 0.5
        if ml > 0:
            return round(100 / (ml + 100), 4)
        else:
            return round(abs(ml) / (abs(ml) + 100), 4)
    except (ValueError, TypeError):
        return float('nan')


def parse_season_table(html: str, season_year: int) -> pd.DataFrame:
    """解析賽季 HTML 表格 → game-level DataFrame"""
    soup = BeautifulSoup(html, 'lxml')
    table = soup.find('table')
    if not table:
        return pd.DataFrame()

    rows = table.find_all('tr')
    header = [th.text.strip() for th in rows[0].find_all(['th', 'td'])]

    data_rows = []
    for row in rows[1:]:
        cells = [td.text.strip() for td in row.find_all(['td', 'th'])]
        if len(cells) >= len(header):
            data_rows.append(cells[:len(header)])

    df_raw = pd.DataFrame(data_rows, columns=header)

    games = []
    # 每兩行：V（客隊）+ H（主隊）
    for i in range(0, len(df_raw) - 1, 2):
        v = df_raw.iloc[i]   # away
        h = df_raw.iloc[i + 1]  # home

        if v.get('VH', '') != 'V' or h.get('VH', '') != 'H':
            continue

        try:
            date_int = int(str(v['Date']).replace('.0', ''))
            month = date_int // 100
            day   = date_int % 100
            year  = season_year - 1 if month >= 10 else season_year
            date_str = f'{year}-{month:02d}-{day:02d}'
        except (ValueError, TypeError):
            date_str = ''

        def safe_int(val):
            try:
                return int(float(val))
            except (ValueError, TypeError):
                return None

        away_pts = safe_int(v.get('Final', ''))
        home_pts = safe_int(h.get('Final', ''))
        if away_pts is None or home_pts is None:
            continue

        away_ml_raw = v.get('ML', '')
        home_ml_raw = h.get('ML', '')

        # 主隊 Open/Close 是讓分；客隊 Open/Close 是大小分
        try:
            open_spread  = float(h.get('Open', 'nan'))
        except ValueError:
            open_spread = float('nan')
        try:
            close_spread = float(h.get('Close', 'nan'))
        except ValueError:
            close_spread = float('nan')
        try:
            open_total   = float(v.get('Open', 'nan'))
        except ValueError:
            open_total = float('nan')
        try:
            close_total  = float(v.get('Close', 'nan'))
        except ValueError:
            close_total = float('nan')

        away_ml_prob = ml_to_prob(away_ml_raw)
        home_ml_prob = ml_to_prob(home_ml_raw)

        # 去除 vig 後的公平勝率
        overround = away_ml_prob + home_ml_prob if (
            not pd.isna(away_ml_prob) and not pd.isna(home_ml_prob) and
            away_ml_prob > 0 and home_ml_prob > 0
        ) else float('nan')

        games.append({
            'date':            date_str,
            'season':          season_year,
            'away_team':       v['Team'],
            'home_team':       h['Team'],
            'away_pts':        away_pts,
            'home_pts':        home_pts,
            'home_win':        1 if home_pts > away_pts else 0,
            'open_spread':     open_spread,   # 主隊讓分（負數 = 主場是熱門）
            'close_spread':    close_spread,
            'open_total':      open_total,
            'close_total':     close_total,
            'away_ml':         away_ml_raw,
            'home_ml':         home_ml_raw,
            'away_ml_prob':    away_ml_prob,  # 含 vig 的隱含勝率
            'home_ml_prob':    home_ml_prob,
            'overround':       round(overround, 4) if not pd.isna(overround) else float('nan'),
            'home_fair_prob':  round(home_ml_prob / overround, 4) if not pd.isna(overround) else float('nan'),
        })

    return pd.DataFrame(games)


def scrape_season(season_year: int, suffix: str) -> pd.DataFrame:
    url = BASE_URL.format(suffix)
    print(f'爬取 {season_year} 賽季（{suffix}）... ', end='', flush=True)

    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        if resp.status_code != 200:
            print(f'HTTP {resp.status_code}')
            return pd.DataFrame()

        df = parse_season_table(resp.text, season_year)
        print(f'{len(df)} 場比賽')
        return df

    except requests.RequestException as e:
        print(f'錯誤：{e}')
        return pd.DataFrame()


if __name__ == '__main__':
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    all_dfs = []

    for season_year, suffix in SEASON_MAP.items():
        df = scrape_season(season_year, suffix)
        if not df.empty:
            all_dfs.append(df)
        time.sleep(2)

    if all_dfs:
        result = pd.concat(all_dfs, ignore_index=True)
        result.to_csv(OUTPUT, index=False, encoding='utf-8-sig')
        print(f'\n✅ 儲存：{OUTPUT}')
        print(f'   共 {len(result)} 場比賽，{result["season"].nunique()} 個賽季')
        print(f'   ML 覆蓋率：{result["home_fair_prob"].notna().mean():.1%}')
        print(result[["date","away_team","home_team","away_pts","home_pts","home_ml","away_ml","home_fair_prob"]].head(5).to_string())
    else:
        print('❌ 未取得任何資料')
