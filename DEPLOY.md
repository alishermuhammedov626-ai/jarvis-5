# JARVIS 5 — serverga o'rnatish va ishga tushirish

Server: `root@46.224.160.2`

> **Muhim:** bu yerdagi barcha buyruqlar **sizning serveringizda** ishlaydi.
> Men serverga ulana olmayman — quyidagilar nusxa-ko'chirib qo'yiladigan
> tayyor buyruqlar.

---

## 1. Serverga ulanish va tizimni tayyorlash

```bash
ssh root@46.224.160.2
```

Debian/Ubuntu uchun:

```bash
apt-get update
apt-get install -y python3 python3-venv python3-pip git curl tmux
python3 -V          # 3.8 yoki undan yuqori bo'lishi kerak
```

---

## 2. Kodni serverga olib borish

**Variant A — git (repo'ga kirish huquqingiz bo'lsa):**

```bash
cd /opt
git clone -b claude/tender-dijkstra-70q69m \
    https://github.com/alishermuhammedov626-ai/jarvis-5.git
cd jarvis-5
```

**Variant B — o'z kompyuteringizdan scp bilan:**

```bash
# LOKAL kompyuteringizda:
cd /path/to/jarvis-5
tar --exclude='.git' --exclude='data' --exclude='results' \
    --exclude='reports' --exclude='.venv' -czf jarvis5.tar.gz .
scp jarvis5.tar.gz root@46.224.160.2:/opt/

# SERVERDA:
ssh root@46.224.160.2
mkdir -p /opt/jarvis-5 && cd /opt/jarvis-5
tar -xzf /opt/jarvis5.tar.gz
```

---

## 3. Preflight — uzoq ishni boshlashdan OLDIN

Bu eng muhim qadam. U 1-2 daqiqada quyidagilarni aytadi: Python versiyasi,
disk joyi, xotira, **Binance API serverdan ochiqmi**, va **har bir coinda
haqiqatda necha kunlik tarix bor**.

```bash
cd /opt/jarvis-5
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/preflight.py --days 365
```

Kutilayotgan natija:

```
  [OK  ] python_version            3.11.x (need >= 3.8)
  [OK  ] disk_space                ... free, need about 394.6MB
  [OK  ] binance_reachable         fapi.binance.com responded in 42ms

  symbol history actually available on the exchange:
    DOGEUSDT           365.0 days
    1000PEPEUSDT       365.0 days
    ...
    PUMPUSDT           128.0 days  <-- only 128 of 365 days requested

 PREFLIGHT: READY
```

`PREFLIGHT: BLOCKED` chiqsa — 6-bo'limga qarang.

---

## 4. To'liq ishga tushirish

Bu **soatlar** davom etadi. SSH uzilib qolsa ish o'lmasligi uchun `tmux`
yoki `nohup` ishlating.

```bash
cd /opt/jarvis-5
tmux new -s jarvis
bash scripts/server_run.sh 365
# ajralish: Ctrl+B keyin D
# qaytish:  tmux attach -t jarvis
```

Yoki:

```bash
nohup bash scripts/server_run.sh 365 > jarvis5_run.log 2>&1 &
tail -f jarvis5_run.log
```

Skript 7 bosqichni ketma-ket bajaradi:

| # | Bosqich | Taxminiy vaqt |
|---|---|---|
| 1 | Python muhiti (venv + requests) | 30 s |
| 2 | Preflight | 1–2 daq |
| 3 | 78 ta test | 1 daq |
| 4 | Research **1-bosqich** — sintetik gate | 30 s |
| 5 | Binance M1 data yuklash (365 kun × 6 coin) | 15–30 daq |
| 6 | Trading backtest (17x, 0.35%) | 6–8 daq |
| 7 | Research **2-bosqich** — real data | 35–45 daq |

Jami taxminan **1–1.5 soat**.

Har bosqich **qaytadan ishga tushirilsa kaldi joyidan davom etadi** —
yuklab olingan candlelar qayta yuklanmaydi.

### Variantlar

```bash
bash scripts/server_run.sh 90               # avval 90 kunlik qisqa sinov
bash scripts/server_run.sh 365 fast         # research ~3x tez (stride 15)
bash scripts/server_run.sh 365 backtest     # faqat trading backtest
bash scripts/server_run.sh 365 research     # faqat research
bash scripts/server_run.sh --help
```

Muhit o'zgaruvchilari:

```bash
COINS=DOGEUSDT,WIFUSDT bash scripts/server_run.sh 365   # faqat 2 coin
SKIP_DOWNLOAD=1 bash scripts/server_run.sh 365          # data allaqachon bor
SKIP_PREFLIGHT=1 ...                                    # preflightni majburan o'tkazish
```

**Tavsiya:** birinchi marta `bash scripts/server_run.sh 90` bilan sinab
ko'ring. Hammasi to'g'ri ishlasa, keyin 365 kunlik to'liq runni qo'ying.

---

## 5. Natijalarni olish

Serverda:

```
reports/FINAL_REPORT.md          <-- BIRINCHI SHUNI O'QING (30 ta savolga javob)
reports/top_setups.csv           eng yaxshi patternlar
reports/oos_summary.csv          discovery / validation / OOS
reports/strategy_by_coin.csv     coin bo'yicha
reports/data_quality.csv         har coinda necha kun data bor
results/jarvis5_365d_*_summary.txt   trading backtest natijasi
results/jarvis5_365d_*.tar.gz        hammasi bitta arxivda
```

Lokal kompyuteringizga:

```bash
scp root@46.224.160.2:/opt/jarvis-5/results/jarvis5_365d_*.tar.gz .
tar -xzf jarvis5_365d_*.tar.gz
```

Yoki faqat asosiy hisobotni:

```bash
scp root@46.224.160.2:/opt/jarvis-5/reports/FINAL_REPORT.md .
```

---

## 6. Muammolar

### `PREFLIGHT: BLOCKED` — binance_reachable FAIL

**Bu eng ehtimoliy muammo.** Binance ba'zi datacenter IP'larini va ba'zi
davlatlarni bloklaydi. Tekshiring:

```bash
curl -sS https://fapi.binance.com/fapi/v1/time
```

Agar javob kelmasa yoki `451` / `403` qaytsa:

- serveringiz IP'si Binance tomonidan bloklangan;
- yechim: boshqa regiondagi VPS (masalan Yaponiya, Singapur, Germaniya),
  yoki o'zingizga tegishli proxy.

Proxy ishlatsangiz, yuklashdan oldin:

```bash
export HTTPS_PROXY=http://user:pass@proxy-host:port
python scripts/preflight.py --days 365
```

Bu faqat **yuklab olish** bosqichiga kerak. Backtest va research
internetsiz ishlaydi — datani boshqa mashinada yuklab, `data/` papkasini
serverga `scp` bilan ko'chirsangiz ham bo'ladi.

### `python3-venv is missing`

```bash
apt-get install -y python3-venv
```

### Disk to'lib qolsa

365 kun × 6 coin ≈ **400 MB** data + hisobotlar. `df -h` bilan tekshiring.

### Xotira yetmasa (2 GB dan kam RAM)

```bash
bash scripts/server_run.sh 365 fast
# yoki yanada kamroq xotira uchun coinlarni bo'lib ishlating:
COINS=DOGEUSDT,1000PEPEUSDT bash scripts/server_run.sh 365
COINS=1000SHIBUSDT,PUMPUSDT bash scripts/server_run.sh 365
```

### PUMPUSDT'da 365 kunlik tarix yo'q

Bu normal — u yaqinda listing qilingan. Preflight aniq necha kun borligini
aytadi, download bor narsani oladi, `data_quality.csv` esa haqiqiy davrni
yozib qo'yadi. Uning sample'i kichikroq bo'ladi — hisobotda uni boshqa
coinlar bilan bir xil deb o'qimang.

---

## 7. Faqat ma'lumot yuklash (backtest'siz)

```bash
source .venv/bin/activate
python -m jarvis5 download --days 365 --data-dir data
du -sh data
```

Keyin istagan vaqtda:

```bash
python -m jarvis5 backtest --days 365 --data-dir data --out results
python -m jarvis5.research.synthetic
python -m jarvis5.research --coins DOGEUSDT,1000PEPEUSDT,1000SHIBUSDT,PUMPUSDT,1000BONKUSDT,WIFUSDT --days 365
```

---

## 8. Xavfsizlik eslatmasi

Bu repo'da **hech qanday API kalit yo'q** va kerak ham emas — faqat
Binance'ning ochiq (public) tarixiy ma'lumot endpointlari ishlatiladi.
Hech qachon bu yerga trading kalitingizni qo'ymang: bu backtest/research
tizimi, u buyurtma yubormaydi.
