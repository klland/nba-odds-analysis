"""
執行順序：
1. 爬取比賽結果
2. 爬取賠率
3. 合併資料
4. 分析 + 產生圖表
"""

import subprocess
import sys
import os

src = os.path.join(os.path.dirname(__file__), 'src')

steps = [
    ('爬取 NBA 比賽結果', os.path.join(src, 'scraper_games.py')),
    ('爬取歷史賠率',      os.path.join(src, 'scraper_odds.py')),
    ('合併資料',          os.path.join(src, 'merge.py')),
    ('分析與視覺化',      os.path.join(src, 'analyze.py')),
]

for i, (name, script) in enumerate(steps, 1):
    print(f'\n{"="*50}')
    print(f'步驟 {i}：{name}')
    print('='*50)
    result = subprocess.run([sys.executable, script], check=False)
    if result.returncode != 0:
        print(f'❌ 步驟 {i} 失敗，停止執行')
        sys.exit(1)

print('\n🎉 全部完成！請查看 output/ 資料夾的圖表')
