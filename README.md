# NBA Odds Analysis

NBA 比賽結果與賠率資料爬取、合併、分析的 Python 專案，產出圖表與統計報告。

## 功能

- **爬取比賽結果** — 自動抓取 NBA 歷史比賽結果
- **爬取歷史賠率** — 爬取各場比賽的博弈賠率資料
- **資料合併** — 將比賽結果與賠率對應合併
- **分析與視覺化** — 統計分析並產生圖表輸出
- **進階分析** — 市場效率、特徵工程等深度分析
- **Jupyter Notebook** — 互動式探索分析

## 專案結構

```
nba-odds-analysis/
├── run.py              # 一鍵執行全流程
├── src/
│   ├── scraper_games.py      # 爬取比賽結果
│   ├── scraper_odds.py       # 爬取歷史賠率
│   ├── merge.py              # 合併資料
│   ├── merge_odds.py         # 賠率合併處理
│   ├── analyze.py            # 基本分析與視覺化
│   ├── analyze_advanced.py   # 進階分析
│   ├── analyze_market.py     # 市場效率分析
│   └── feature_engineering.py # 特徵工程
├── notebooks/
│   └── NBA_Analysis.ipynb    # Jupyter 互動分析
├── data/               # 原始資料
└── output/             # 分析結果與圖表
```

## 使用方式

### 一鍵執行全流程

```bash
python run.py
```

依序執行：爬取比賽結果 → 爬取賠率 → 合併資料 → 分析與視覺化

### 單獨執行各步驟

```bash
python src/scraper_games.py   # 爬取比賽結果
python src/scraper_odds.py    # 爬取賠率
python src/merge.py           # 合併資料
python src/analyze.py         # 分析
```

### Jupyter Notebook

```bash
jupyter notebook notebooks/NBA_Analysis.ipynb
```

## 技術

- Python 3
- requests / BeautifulSoup（爬蟲）
- pandas（資料處理）
- matplotlib / seaborn（視覺化）
- Jupyter Notebook（互動分析）
