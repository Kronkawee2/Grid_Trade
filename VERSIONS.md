# โครงสร้างโปรเจกต์

**baseline คือฐานเทียบ ล็อกแล้ว ไม่แตะ** ส่วนงานพัฒนาไล่เป็น v1, v2, v3 ไปเรื่อยๆ

| | baseline | v1_highfreq |
| :--- | :--- | :--- |
| สถานะ | **ล็อก ห้ามแก้** | งานพัฒนาชิ้นที่ 1 |
| ค่า | `configs/baseline.json` | `configs/v1_highfreq.json` |
| โค้ดรัน | `scripts/baseline/` | `scripts/v1_highfreq/` |
| ผล | `results/baseline/` | `results/v1_highfreq/` |
| กราฟ | `plots/baseline/` | `plots/v1_highfreq/` |

งานพัฒนาชิ้นถัดไปตั้งชื่อ `v2_xxx` แล้วสร้างครบทั้งสี่โฟลเดอร์ **ไม่ทับของเดิม**

## เครื่องยนต์ใช้ร่วมกัน — และต้องเป็นแบบนั้น

`scripts/engine/adaptive.py` **ถ้าแยกกัน จะเทียบผลกันไม่ได้เลย** เพราะจะไม่รู้ว่า
ตัวเลขที่ต่างกันมาจากค่าพารามิเตอร์ หรือมาจากโค้ดคนละตัว

แก้ engine เมื่อไหร่ ต้องรัน `scripts/engine/test_adaptive.py` ให้ผ่านครบ 24 ข้อ แล้วรันใหม่ทุกเวอร์ชัน

## วิธีรัน

```
python scripts/engine/test_adaptive.py        # เทสต์เครื่องยนต์ ต้องผ่านก่อน

python scripts/baseline/plots_current.py      # -> plots/baseline/
python scripts/baseline/yearly_results.py     # -> results/baseline/
python scripts/baseline/stagnation.py

python scripts/v1_highfreq/plots_current.py   # -> plots/v1_highfreq/
python scripts/v1_highfreq/yearly_results.py
python scripts/v1_highfreq/stagnation.py
```

แต่ละสคริปต์ล็อกชื่อเวอร์ชันของตัวเองไว้ในไฟล์ ไม่ต้องส่งพารามิเตอร์

## โฟลเดอร์อื่น

| | |
| :--- | :--- |
| `scripts/engine/` | เครื่องยนต์ + เทสต์ 24 ข้อ |
| `scripts/exploration/` | งานค้นหาครั้งเดียว ไม่ผูกกับเวอร์ชันไหน |
| `_invalid_outputs/` | ของที่สร้างก่อนแก้บั๊ก **ห้ามอ้างเป็นผลลัพธ์** |
| `wiki_PNG/` | ของเจ้าของโปรเจกต์ ไม่มีสคริปต์ไหนเขียนลงไป |

## ผล (2006-2024, m5, lot 0.01)

| | baseline | v1_highfreq |
| :--- | ---: | ---: |
| EURUSD | $1,333 · DD 16% · เดือน+ 62% | **$2,425 · DD 13% · เดือน+ 89%** |
| XAUUSD | $2,238 · DD 18% · เดือน+ 67% | **$4,015 · DD 16% · เดือน+ 92%** |
| ไม้/ปี | 65 | 290 |
| ผ่าน 3 ด่าน | ครบ | ผ่าน 2 ตก 1 (ด่านอันดับ) |
