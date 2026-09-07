# เวอร์ชันในโปรเจกต์นี้

**แยกขาดจากกันทั้งค่า โค้ด ผล และกราฟ** แก้ v2 ไม่มีทางกระทบ v1

| | v1_baseline | v2_highfreq |
| :--- | :--- | :--- |
| สถานะ | **ล็อกแล้ว ห้ามแก้** | งานพัฒนา แก้ได้ |
| ค่า | `configs/v1_baseline.json` | `configs/v2_highfreq.json` |
| โค้ดรัน | `scripts/v1_baseline/` | `scripts/v2_highfreq/` |
| ผล | `results/v1_baseline/` | `results/v2_highfreq/` |
| กราฟ | `plots/v1_baseline/` | `plots/v2_highfreq/` |

## เครื่องยนต์ใช้ร่วมกัน — และต้องเป็นแบบนั้น

`scripts/engine/adaptive.py` ใช้ร่วมกันทั้งสองเวอร์ชัน **ถ้าแยกกัน จะเทียบผลกันไม่ได้เลย**
เพราะจะไม่รู้ว่าตัวเลขที่ต่างกันมาจากค่าพารามิเตอร์ หรือมาจากโค้ดคนละตัว

แก้ engine เมื่อไหร่ ต้องรัน `scripts/engine/test_adaptive.py` ให้ผ่านครบ แล้วรันใหม่ทั้งสองเวอร์ชัน

## วิธีรัน

```
python scripts/engine/test_adaptive.py          # เทสต์เครื่องยนต์ ต้องผ่านก่อน

python scripts/v1_baseline/plots_current.py     # -> plots/v1_baseline/
python scripts/v1_baseline/yearly_results.py    # -> results/v1_baseline/
python scripts/v1_baseline/stagnation.py

python scripts/v2_highfreq/plots_current.py     # -> plots/v2_highfreq/
python scripts/v2_highfreq/yearly_results.py
python scripts/v2_highfreq/stagnation.py
```

แต่ละสคริปต์ล็อกชื่อ config ของตัวเองไว้ในไฟล์ ไม่ต้องส่งพารามิเตอร์

## โฟลเดอร์อื่น

| | |
| :--- | :--- |
| `scripts/engine/` | เครื่องยนต์ + เทสต์ 24 ข้อ ใช้ร่วมกัน |
| `scripts/exploration/` | งานค้นหาครั้งเดียว ไม่ผูกกับเวอร์ชันไหน |
| `results/exploration/` | ผลของงานพวกนั้น |
| `_invalid_outputs/` | ของที่สร้างก่อนแก้บั๊ก **ห้ามอ้างเป็นผลลัพธ์** |
| `wiki_PNG/` | ของเจ้าของโปรเจกต์ ไม่มีสคริปต์ไหนเขียนลงไป |

## ผลปัจจุบัน (2006-2024, m5, lot 0.01)

| | v1_baseline | v2_highfreq |
| :--- | ---: | ---: |
| EURUSD | $1,333 · DD 16% · เดือน+ 62% | **$2,425 · DD 13% · เดือน+ 89%** |
| XAUUSD | $2,238 · DD 18% · เดือน+ 67% | **$4,015 · DD 16% · เดือน+ 92%** |
| ไม้/ปี | 65 | 290 |
| ผ่าน 3 ด่าน | ครบ | ผ่าน 2 ตก 1 (ด่านอันดับ) |
