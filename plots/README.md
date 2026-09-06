# กราฟในโปรเจกต์นี้

## `baseline/` — ค่าที่ใช้อยู่จริง

ทุกรูปสร้างจาก `params_100usd.json` โดยตรง: **lot 0.01 คงที่ ไม่ทบต้น
ไม่ตัดช่วงเวลาใดๆ** ตัวเลขในรูปเหล่านี้คือตัวเลขเดียวกับที่รายงานใน README

| ไฟล์ | แสดงอะไร |
| :--- | :--- |
| `account_EURUSD_train.png` | ยอดเงิน + drawdown + กำไรรายเดือน · EURUSD 2006-2024 |
| `account_EURUSD_oos.png` | เหมือนกัน · ช่วง 2025-2026 |
| `account_GOLD_train.png` | ทองคำ 2006-2024 |
| `account_GOLD_oos.png` | ทองคำ 2025-2026 |
| `grid_*.png` | ราคา + กล่องกริด + จุดเข้าออก + แผง volatility + equity (4 ใบ) |
| `regime_by_year.png` | กำไรรายปี เทียบกับความผันผวนของปีนั้น |
| `stagnation_baseline.png` | ช่วงที่บัญชีไม่ทำจุดสูงสุดใหม่ (แถบชมพู) |

## `experiments/` — ข้อเสนอที่ยังไม่ได้ใช้

**ไม่ใช่ baseline** อย่าเอาตัวเลขในนี้ไปอ้างเป็นผลของระบบ

| ไฟล์ | แสดงอะไร |
| :--- | :--- |
| `session_filter_test.png` | เทียบกับการตัดช่วงเอเชีย และตัดเอเชีย+ชั่วโมงนิ่ง |
| `stagnation_compounding.png` | ช่วงนิ่งเมื่อเปิดการทบต้น (ตัวเลขทบต้นยังมีบั๊ก ดู README) |

## สร้างใหม่

```
python scripts/backtest/plots_current.py
python scripts/backtest/stagnation.py
python scripts/backtest/stagnation.py --compound   # -> experiments/
```

`wiki_PNG/` เป็นของเจ้าของโปรเจกต์ ไม่มีสคริปต์ไหนเขียนลงไป
