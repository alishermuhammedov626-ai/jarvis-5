# JARVIS 5

**Binance Futures — SMC + Liquidity + Internal/External Structure scalping backtest system**

M15 → M5 → M1 · 17x · **100 % deterministic rule-based · MACHINE LEARNING YO'Q**

---

## 0. Bu nima

JARVIS 5 — Binance USDT-M Futures uchun **backtest engine**. U narxni bashorat
qilmaydi; bozor allaqachon chop etgan strukturaviy ma'lumot (liquidity, sweep,
MSS/CHOCH/BOS, displacement, OB/FVG) ustidan **qat'iy qoidalar** bo'yicha qaror
qabul qiladi.

Bir xil data + bir xil config = **har doim bir xil natija**. Bu shunchaki
va'da emas, `tests/test_no_lookahead.py` da tekshiriladi.

**Bu versiyada quyidagilar YO'Q va bo'lishi ham mumkin emas:**
Machine Learning, AI, neural network, XGBoost, Random Forest, Logistic
Regression, probability model, ML/AI score, hidden scoring, model-based filter.
`tests/test_rules.py::TestNoMachineLearning` butun paketni skanerlab, shu
konstruksiyalardan birortasi paydo bo'lsa testni yiqitadi.

---

## 1. Tez boshlash (server)

```bash
git clone <repo> && cd jarvis-5
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # faqat `requests`

# 1) Tarixiy datani yuklab olish (faqat shu qadam internet talab qiladi)
python -m jarvis5 download --days 365 --data-dir data

# 2) Asosiy backtest (17x, 6 coin, 0.35 % risk)
python -m jarvis5 backtest --days 365 --data-dir data --out results

# 3) To'liq tekshiruv: backtest + walk-forward + Monte Carlo +
#    robustness + parameter sensitivity + production gate
python -m jarvis5 full --days 365 --data-dir data --out results
```

Internetsiz o'rnatishni tekshirish uchun:

```bash
python -m jarvis5 selftest --out results        # sintetik data, pipeline testi
python -m unittest discover -s tests            # 32 ta test
```

> `selftest` va `scripts/make_synthetic_data.py` **sun'iy narx** ishlatadi.
> Ular faqat o'rnatish to'g'ri ishlayotganini ko'rsatadi — ulardan chiqqan
> hech qanday raqam savdo dalili emas.

### Server uchun tayyor skript

```bash
bash scripts/run_server_backtest.sh 365        # download + full + arxiv
```

---

## 2. Strategiya oqimi

```
M15 context  →  M15 external/internal structure  →  M15 liquidity
      ↓
M5 liquidity sweep  →  M5 MSS / CHOCH / BOS  →  M5 displacement  →  OB / FVG
      ↓
correction / zone retest
      ↓
M1 micro sweep  →  M1 CHOCH / BOS  →  M1 displacement
      ↓
risk check → cost check → liquidation check → ENTRY
      ↓
structure-based SL · liquidity-based TP · M1 structure trailing → EXIT
```

**M15** faqat context beradi — yakka o'zi trade trigger emas (spec 16).
**M5** — setup formationning asosiy timeframe'i. **M1** yo'nalish tanlamaydi,
faqat *qachon* kirishni hal qiladi.

### Setup turlari

| Kod | Nomi | Qachon |
|-----|------|--------|
| `A_REVERSAL` | Reversal | M15 regime setup yo'nalishiga qarshi |
| `B_RANGE_REVERSAL` | Range reversal | M15 RANGE holatida |
| `C_TREND_CONTINUATION` | Trend continuation | M15 regime setup bilan bir xil |
| `D_DOUBLE_LIQUIDITY` | Double liquidity | M15 sweep + M5 sweep + MSS — **high priority** |

---

## 3. No-look-ahead kafolati

Bu tizimning eng muhim qismi. Backtest live tradingni simulyatsiya qiladi:

| Qoida | Amalga oshirilishi |
|-------|--------------------|
| Yopilmagan candle ishlatilmaydi | M5/M15 bar faqat `close_time <= hozirgi M1 close` bo'lsa engine'ga beriladi |
| Swing kelajakni bilmaydi | K=2 fraktal **aynan 2 bar kechikib** e'lon qilinadi |
| Qisman aggregation yo'q | `aggregate()` faqat 5/15 ta M1 candle to'liq bo'lsa bar chiqaradi |
| BOS/CHOCH wick bilan emas | Faqat **close** darajadan `MIN_BREAK_ATR × ATR` uzoqqa o'tsa |
| Trailing o'z barida ishlamaydi | Bar *i* close'ida ko'chirilgan SL eng erta bar *i+1* da tegadi |
| Entry confirmation close'ida | Entry bari o'zida hech qachon exit bo'lmaydi |
| ATR kelajaksiz | Wilder ATR incremental, faqat `<= i` barlardan |

**Truncation invariance testi** — eng kuchli dalil: datani `T` vaqtida kesib
tashlaymiz; `T` gacha yopilgan **barcha** trade'lar bir baytga qadar bir xil
bo'lishi shart. Agar engine bitta kelajak barga qarasa, bu test yiqiladi.

Bundan tashqari **scale invariance** testi: barcha threshold ATR bilan
normalizatsiya qilinganligi sababli, narxni 50x ko'paytirsak trade'lar aynan
bir xil chiqishi kerak — chiqadi.

---

## 4. Risk va execution modeli

### Position size — leverage bilan emas, **risk** bilan

```
RiskAmount       = Equity × RISK_PER_TRADE          10 000 × 0.35 % = $35
PositionNotional = RiskAmount / SL_fraction         35 / 0.0040     = $8 750
Quantity         = Notional / entry   → step size bo'yicha yaxlitlanadi
RequiredMargin   = Notional / LEVERAGE              8 750 / 17      = $514.71
```

Yaxlitlashdan **keyin** real risk qayta hisoblanadi va qayta tekshiriladi —
jonli akkaunt aynan yaxlitlangan miqdorni ko'taradi.

17x faqat margin / liquidation / execution mexanikasiga ta'sir qiladi. U
akkaunt riskini 17 barobar qilmaydi.

### Stop loss — structure-based

`BUY: SL = invalidation_low − 0.10 × ATR` (M1 swing low / sweep low / zone floor).
M1 struktura juda tor yoki keng bo'lsa — M5 invalidation swing'ga fallback.
`0.30 % ≤ SL ≤ 0.50 %` chegarasidan chiqsa → `SL_TOO_TIGHT` / `SL_TOO_WIDE`.

### Take profit — liquidity-based

Hech qachon "fixed candle target" emas. Yo'nalishga qarshi turgan eng yaqin
**tegilmagan** liquidity, `MIN_TP_PERCENT` (1.0 %) dan uzoqda bo'lgani
tanlanadi. Topilmasa → `TARGET_TOO_CLOSE`.

RR **hard filter emas** (spec 33) — u faqat log qilinadi va statistikada
ko'rsatiladi, chunki uning foydasi hali OOS bilan tasdiqlanmagan.

### Costs — natijani bo'yamaslik uchun

* Commission **notional** ustidan hisoblanadi, margin ustidan emas
  (17x da farq 17 barobar). Ikkala tomon uchun.
* Slippage har tomonga `SLIPPAGE_BPS`, **doim noqulay** tomonga.
* Funding tarixi bo'yicha; long musbat rate'da to'laydi, short oladi.
* **Funding data yo'q bo'lsa 0 deb yashirilmaydi** — hisobotda
  `FUNDING DATA UNAVAILABLE` deb ko'rsatiladi.
* Natija 4 qatlamda beriladi: `without cost` → `+commission` →
  `+slippage` → `+all costs`.

### Bir candle ichida SL va TP ikkalasi tegsa

M1 ichidagi tick tartibi **bizga noma'lum**. Default `SAME_CANDLE_RULE=SL_FIRST`
— eng konservativ variant, va bunday trade `ambiguous_exit` deb belgilanadi
hamda hisobotda alohida sanaladi. Gap bo'lsa fill **open** narxida bo'ladi.

---

## 5. Trade journal — over-filtering'ni ko'rish uchun

Faqat trade'lar emas, **har bir candidate** yoziladi. Candidate M5 liquidity
sweep paydo bo'lganda tug'iladi va o'z natijasigacha kuzatiladi:
`FOUND` yoki `REJECTED` + aniq sabab.

`primary_rejection_reason` va `secondary_rejection_reasons` alohida saqlanadi.
Hisobotdagi **CANDIDATE FUNNEL** jadvali — bu strategiyaning o'zini
bo'g'ib qo'ymaganini tekshiradigan asosiy asbob. Agar massaning 90 % i bitta
sababda tursa, bu strategiya natijasi emas, dizayn xatosi.

Rejection sabablari: `NO_M15_CONTEXT`, `NO_LIQUIDITY`, `NO_M5_SWEEP`, `NO_MSS`,
`NO_CHOCH`, `NO_BOS`, `NO_DISPLACEMENT`, `NO_OB`, `NO_FVG`, `NO_ZONE`,
`NO_RETEST`, `NO_M1_CONFIRMATION`, `TARGET_TOO_CLOSE`, `SL_TOO_WIDE`,
`SL_TOO_TIGHT`, `SL_IMPOSSIBLE`, `CHASE`, `LIQUIDATION_RISK`, `COST_TOO_HIGH`,
`DUPLICATE`, `DAILY_RISK_LIMIT`, `DAILY_TRADE_LIMIT`, `ACTIVE_POSITION`,
`MAX_POSITIONS`, `SETUP_EXPIRED`, `SETUP_INVALIDATED`, `INVALID_STRUCTURE`,
`INVALID_DATA`, `PREMIUM_DISCOUNT`, `SESSION_FILTER`, `COOLDOWN`,
`INVALID_SIZE`, `RR_TOO_LOW`.

---

## 6. Chiqadigan fayllar (`results/`)

| Fayl | Ichida |
|------|--------|
| `*_summary.txt` | To'liq matnli hisobot (pastdagi barcha jadvallar) |
| `*_report.json` | O'sha hisobot mashina o'qiy oladigan ko'rinishda |
| `*_trades.csv` | Har trade uchun ~80 ustun: M15/M5/M1 konteksti, sweep, zone, risk, cost, MFE/MAE |
| `*_equity.csv` | Portfel equity + drawdown curve |
| `*_symbol_equity.csv` | Har coin uchun alohida equity curve |
| `*_candidates.csv` | Har bir candidate va uning rejection sababi |
| `*_states.csv` | State machine tranzitsiyalari |
| `*_analysis.json` | walk-forward + Monte Carlo + robustness + sensitivity + gate (`full`) |

Hisobotdagi kesimlar: per coin, per setup (A/B/C/D), long/short, per session
(Asia/London/NY/OFF), per market regime, direction × regime, per liquidity
type, per sweep quality, per sweep penetration, per displacement bucket, per
zone type (fresh/mitigated), per structure event, per M1 confirmation
kombinatsiyasi, per month, per exit reason.

Har jadvalda **win rate yolg'iz ko'rsatilmaydi** — yonida doim PF,
expectancy R, avg R va max DD turadi.

---

## 7. Validatsiya asboblari

| Buyruq | Nima qiladi |
|--------|-------------|
| `walkforward` | Xronologik TRAIN 60 % / VALIDATION 20 % / OOS 20 %. Random shuffle **yo'q**. Rolling rejimi ham bor. |
| `montecarlo` (`full` ichida) | Yopilgan trade R ketma-ketligini 5 000 marta resample qiladi: final equity taqsimoti, max DD taqsimoti, losing streak, risk of ruin |
| `sensitivity` | 13 ta parametr × (−20 %, −10 %, 0, +10 %, +20 %). Faqat markazda ishlasa → `OVERFIT_RISK` |
| `robustness` | commission +10/+20 %, slippage +10/+20 %, 1-candle entry delay, TP_FIRST, trailing o'chirilgan, risk 0.25/0.50 %. Ko'pi yiqilsa → `FRAGILE` |
| `full` ichidagi **production gate** | 12 mezon: OOS expectancy, PF > 1, DD nazorati, trade soni, cost-inclusive foyda, coinlar bo'yicha barqarorlik, davrlar bo'yicha barqarorlik, parametr sezgirligi, execution robustness, no look-ahead, realistic execution, no ML |

Production gate **hech qachon** faqat win rate bilan ochilmaydi. 70 % WR
bo'lsa ham PF ≤ 1 bo'lsa — strategiya yutqazadi.

---

## 8. Konfiguratsiya

Barcha 83 parametr `jarvis5/config.py` da; hech bir threshold kod ichiga
qotirilmagan.

```bash
python -m jarvis5 config --write config/my.json          # joriy configni yozish
python -m jarvis5 backtest --config config/my.json --days 365
python -m jarvis5 backtest --set RISK_PER_TRADE=0.005 --set LEVERAGE=10
python -m jarvis5 backtest --symbols DOGEUSDT,WIFUSDT --days 90
```

Tayyor presetlar: `config/default.json`, `risk_025.json`, `risk_050.json`,
`loose_m1.json` (M1 stack qanchaga tushayotganini o'lchash uchun),
`strict.json`.

Asosiy parametrlar:

| Parametr | Default | Ma'nosi |
|----------|---------|---------|
| `SYMBOLS` | 6 coin | DOGE, 1000PEPE, 1000SHIB, PUMP, 1000BONK, WIF |
| `LEVERAGE` | 17 | faqat margin/liquidation mexanikasi |
| `RISK_PER_TRADE` | 0.0035 | ruxsat: 0.0025 / 0.0035 / 0.0050 |
| `MAX_DAILY_LOSS` | 0.015 | realized net −1.5 % → kun yopiladi |
| `MAX_TRADES_PER_SYMBOL_PER_DAY` | 3 | 6 × 3 = kuniga maks 18 (majburiy emas) |
| `ATR_PERIOD` / `SWING_K` | 14 / 2 | |
| `EXTERNAL_STRUCTURE_ATR` | 1.5 | external swing separation |
| `EQUAL_LIQUIDITY_ATR` / `LIQUIDITY_CLUSTER_ATR` | 0.10 / 0.25 | equal H/L va cluster |
| `MIN_DISPLACEMENT_ATR` / `STRONG_DISPLACEMENT_ATR` | 1.0 / 1.5 | |
| `MIN_FVG_ATR` | 0.05 | |
| `MAX_CHASE_ATR` | 0.50 | zonadan uzoqlashsa entry yo'q |
| `MIN_SL_PERCENT` / `MAX_SL_PERCENT` | 0.0030 / 0.0050 | |
| `STRUCTURAL_BUFFER_ATR` | 0.10 | |
| `MIN_TP_PERCENT` | 0.010 | |
| `TRAILING_ACTIVATION_PERCENT` | 0.0035 | M1 HL/LH strukturasi bo'yicha |
| `COOLDOWN_MINUTES` | 15 | |
| `MAX_HOLD_HOURS` | 24 | → `TIMEOUT` |
| `COMMISSION_RATE` / `SLIPPAGE_BPS` | 0.0004 / 2.0 | har tomonga |
| `SAME_CANDLE_RULE` | `SL_FIRST` | konservativ |

Soft (hard emas) filtrlar, spec talabiga ko'ra: premium/discount
(`PD_HARD_FILTER=False`), session (`SESSION_FILTER=[]`), RR
(`RR_HARD_FILTER=False`), sweep quality (`ALLOW_WEAK_SWEEP=True`). Ularning
har biri statistikada alohida kesim sifatida chiqadi — avval o'lchang, keyin
hard filterga aylantiring, va faqat OOS tasdiqlasa.

---

## 9. Kod tuzilishi

```
jarvis5/
  config.py                83 parametr, JSON save/load, validate
  core/series.py           array.array ustidagi OHLCV, M1→M5/M15 aggregation
  core/timeutil.py         UTC ms, trading day, Tashkent sessiyalari, funding vaqti
  engine/atr.py            incremental Wilder ATR
  engine/swings.py         K-fraktal, kechiktirilgan tasdiq
  engine/structure.py      internal/external leg, HH/HL/LH/LL, BOS/CHOCH/MSS, regime, PD
  engine/liquidity.py      10 xil level, equal H/L cluster, sweep + quality scoring
  engine/displacement.py   ATR-normalizatsiyalangan impuls
  engine/zones.py          OB, FVG, OB+FVG confluence, freshness
  engine/timeframe.py      bitta timeframe'ning to'liq holati
  strategy/setups.py       M15 context → M5 setup (A/B/C/D), state machine
  strategy/confirmation.py zone retest + M1 micro stack + chase protection
  strategy/risk.py         SL/TP/size/liquidation/cost pre-check
  strategy/journal.py      candidate journal + rejection taksonomiyasi
  backtest/engine.py       portfel backtest, umumiy equity va limitlar
  backtest/position.py     trailing, exit, MFE/MAE, same-candle qoidasi
  backtest/costs.py        commission / slippage / funding
  backtest/metrics.py      barcha statistika va kesimlar
  backtest/report.py       txt / json / csv hisobotlar
  analysis/                walkforward, montecarlo, sensitivity, robustness, gate
  data/binance.py          kline + funding + exchangeInfo yuklovchi va cache
  data/synthetic.py        offline tekshiruv uchun deterministik sun'iy data
```

---

## 10. Ishlash tezligi

Sof stdlib, `array.array` ustida. Taxminan **~9 000 M1 candle/sekund**:

| Hajm | Vaqt |
|------|------|
| 6 coin × 90 kun | ~1.5 daqiqa |
| 6 coin × 365 kun | ~6–7 daqiqa |
| `full` (365 kun, sensitivity bilan) | ~6 soat — kechasi `nohup` bilan qo'ying |
| `full --skip-sensitivity` | ~40 daqiqa |

RAM: 1 yillik 6 coin ≈ 250 MB.

---

## 11. Muhim eslatmalar

* **Backtest natijasi kelajak kafolati emas.** Monte Carlo ham bashorat emas —
  u faqat "shu trade taqsimoti bilan omad qanchalik yomon bo'lishi mumkin"
  degan savolga javob beradi.
* **70 % WR maqsad bo'lishi mumkin, lekin kafolat sifatida yozilmaydi.**
  Hisobot WR yonida doim PF va expectancy'ni ko'rsatadi.
* Bitta backtest natijasiga qarab threshold tanlamang. `sensitivity`
  `STABLE` bermaguncha, tanlangan qiymat curve fit bo'lishi mumkin.
* Strategiya faqat cost-free holatda foydali bo'lsa — u production-ready emas.
  Shuning uchun hisobot 4 qatlamli cost breakdown beradi.
* Bu repo **backtest** tizimi. Live order routing, exchange kalitlari va real
  buyurtma yuborish bu yerda **yo'q**.

---

# JARVIS 5 RESEARCH — Market Move Cause Discovery

> **Bu modul trading engine'ni O'ZGARTIRMAYDI.** U alohida `jarvis5/research/`
> paketida yashaydi, faqat mavjud detektorlarni **o'qish uchun** ishlatadi va
> hech qanday yangi strategiya, ML, AI yoki probability score qo'shmaydi.
> `tests/test_research.py::TestTradingEngineUntouched` buni tekshiradi.

## Asosiy savol

> "6 ta coinda 1 yil davomida yuz bergan **1%+ harakatlar oldidan** bozor
> qanday pattern va kombinatsiyalar asosida harakat qilgan?"

## Ikki bosqich — real dataga avtomatik o'tilmaydi

```bash
# BOSQICH 1 — sintetik validatsiya (real dataga tegmaydi)
python -m jarvis5.research.synthetic

# BOSQICH 2 — real Binance data (1-bosqich PASS bo'lmasa ISHLAMAYDI)
python -m jarvis5.research \
  --coins DOGEUSDT,1000PEPEUSDT,1000SHIBUSDT,PUMPUSDT,1000BONKUSDT,WIFUSDT \
  --days 365 --timeframe 1m
```

Yoki hammasi bitta buyruqda: `bash scripts/run_research.sh 365`

Bosqich 1 muvaffaqiyatli bo'lsa ham **avtomatik ravishda** bosqich 2 ga
o'tmaydi — real dataga qaratish ongli qaror bo'lishi kerak.

## Sintetik validatsiya nima qiladi

10 xil ssenariy **ataylab** yaratiladi va ularning aniq indekslari ground
truth sifatida saqlanadi: sweep→MSS→displacement, sweep→FVG, Wyckoff spring,
RSI reversal, volume breakout, false breakout, trend continuation, range
breakout, **RANDOM** va **NOISE** (pattern shakli bor, lekin davomi yo'q).

Tekshiriladi: event coverage, pattern detection, kombinatsiya yig'ilishi,
MFE/MAE izchilligi, va eng muhimi — o'sha detektorlar RANDOM segmentda ham
xuddi shunday tez-tez ishlaydimi. **Har joyda "sweep" deb baqiradigan
detektor hech nima topmagan.**

## Metodologiya — nega bu "win rate" emas

Asosiy ko'rsatkich **edge**:

```
edge = P(1% move | pattern) − P(1% move | pattern yo'q)
```

Control group — qolgan **barcha** sampled barlar (tasodifiy tanlanmagan,
balki to'liq). Agar pattern 1% harakatlar oldidan 70% uchrasa, lekin oddiy
bozorda ham 65% uchrasa — bu **5 punktlik edge**, 70 emas. `relative_lift`
va `odds_ratio` ham beriladi.

Baseline populyatsiya har `BASELINE_STRIDE`-bar (default 5) sistematik
tanlanadi va **natijadan mustaqil**. Event origin barlari populyatsiyaga
qo'shilmaydi — aks holda taqqoslash buzilardi.

## No-look-ahead — ikki bosqichli ajratish

| Bosqich | Kelajakka qarashi | Nima qiladi |
|---|---|---|
| **PASS 1 — outcomes** | **Ha, ataylab** | "1% harakat bo'ldimi?" — bu savolga javob berish uchun kelajak kerak |
| **PASS 2 — patterns** | **Yo'q, hech qachon** | Bar-bar oldinga yuradi, faqat o'sha lahzada ko'ringan narsani yozadi |

Ikkalasi faqat **bar indeksi** orqali bog'lanadi. Pattern hech qachon o'zi
oldindan aytishi kerak bo'lgan natijadan hisoblanmaydi.

Tekshiruvlar: **truncation invariance** (datani `T` da kessak, `T` gacha
bo'lgan patternlar bir xil qolishi shart) va **scale invariance** (narx
×10 / ×50 / ×100 — pattern to'plami o'zgarmasligi shart).

## Event aniqlash

Har M1 candle potensial origin. `+1%` (long) yoki `-1%` (short) targetlar
`EVENT_HORIZON_BARS` (default 120 daqiqa) ichida tekshiriladi. Naive skanerlash
O(n×horizon) bo'lardi; bu yerda har target uchun bitta **heap** ishlatiladi —
har origin **birinchi** teginishda hal bo'ladi, umumiy O(n log n).

Bitta impuls yuzlab signal bo'lib sanalmasligi uchun **clustering**: yangi
origin joriy impulsga qo'shiladi, agar impuls hali davom etayotgan bo'lsa
va narx `CLUSTER_RESET_PCT` (50%) qaytmagan bo'lsa. Reset testi faqat
harakat `CLUSTER_MIN_EXTENSION_PCT` (0.5%) uzayganidan keyin yoqiladi —
aks holda range ichidagi oddiy tebranish bitta impulsni o'nlab soxta
eventga bo'lib yuborardi.

Har event **impulse_start** bilan ham belgilanadi: `event_start` va
ekstremum orasidagi eng past low (long uchun). Pattern aynan o'sha barda
o'qiladi — trader haqiqatan harakat qilishi kerak bo'lgan bar.

## O'lchanadigan patternlar

- **SMC**: 10 xil liquidity turi, equal high/low cluster, sweep (+ STRONG/
  MEDIUM/WEAK sifat bahosi), HH/HL/LH/LL, BOS/CHOCH/MSS (internal va
  external alohida), displacement, OB, FVG, breaker, mitigation zone,
  premium/discount/equilibrium — **M1, M5, M15 uchun alohida**
- **Wyckoff**: accumulation, distribution, spring, upthrust, SOS, SOW,
  LPS, LPSY, trading range, breakout/breakdown, false breakout,
  re-accumulation, re-distribution
- **RSI(14)**: <20, <30, 30/40/50 reclaim, >50, >70, >80, bullish/bearish
  divergence, failure swing, momentum reversal
- **Volume**: spike, expansion, contraction, climax, >1.5x/2x/3x,
  high-volume rejection/breakout, low-volume breakout
- **Price action**: engulfing, pin bar, hammer, shooting star, inside/outside
  bar, large body, breakout candle, false breakout, breakout+retest,
  momentum va reversal candle
- **20 ta kombinatsiya** (sweep+BOS, sweep+MSS+FVG+OB, double sweep, ...)
  va **JARVIS setup oilalari A/B/C/D**

Kombinatsiyalarda **tartib muhim**: sweep'dan *oldin* bo'lgan break
"sweep+break" hisoblanmaydi.

Har pattern 6 ta oynada o'lchanadi: **5 / 10 / 20 / 30 / 60 / 120 daqiqa**.
Ichkarida ular *disjoint* yosh binlariga yoziladi va hisobotda kumulyativ
yig'iladi — bir xil arifmetika, 4 barobar tez.

## Halollik qoidalari (spec 15, 16, 26, 35)

- `n < 30` → **INSUFFICIENT SAMPLE**. `n < 100` → **EXPLORATORY**.
- Hech qanday strategiya **100 taga to'ldirilmaydi**. 17 ta bo'lsa — 17 deb yoziladi.
- Bir eventda bir nechta pattern bo'lsa, **hammasi** alohida tag sifatida saqlanadi.
- Verdikt: `ROBUST` / `PROMISING` / `WEAK` / `INSUFFICIENT SAMPLE` /
  `OVERFIT RISK` / `OVERFIT / FAILED OOS`.
- Coin consistency: `CROSS_COIN_CONSISTENT` / `PARTIAL` / `COIN_SPECIFIC` /
  `SINGLE_COIN`. **Bitta coin dalil emas.**
- DISCOVERY 70% → VALIDATION 15% → OOS 15%, faqat xronologik, shuffle yo'q.
- Agar hech bir pattern barcha filtrlardan o'tmasa, savol 30 ga javob
  **"None"** bo'ladi. Ro'yxatni to'ldirish uchun hech nima ko'tarilmaydi.

## Entry simulation (research only)

Har sampled barda gipotetik entry: structure-based SL (0.30–0.50%),
targetlar 1.0 / 1.5 / 2.0 / 3.0%, risk 0.35%, 17x. **WITHOUT TRAILING** va
**WITH TRAILING** (+0.35% activation) alohida. Commission notional ustidan,
slippage har tomonga, funding — mavjud bo'lmasa `UNAVAILABLE`.

Bir xil trade'lar barcha patternlarga taqsimlanadi, shuning uchun patternlar
bir-biri bilan **aynan bir xil savdolarda** taqqoslanadi.

> Bu **strategiya backtesti emas**: kunlik limit, cooldown, setup gating
> yo'q. U patternning qiymatini o'lchaydi, JARVIS 5 nima savdo qilishini emas.

## Chiqadigan fayllar (`reports/`)

`overall_summary.csv` · `coin_summary.csv` · `strategy_summary.csv` ·
`strategy_by_coin.csv` · `strategy_by_direction.csv` ·
`combination_summary.csv` · `event_summary.csv` · `oos_summary.csv` ·
`rejection_summary.csv` · `data_quality.csv` · `top_setups.csv` ·
`full_event_database.csv` · `full_report.json` · **`FINAL_REPORT.md`**

`FINAL_REPORT.md` — 30 ta savolga to'liq javob (spec 28).

## Konfiguratsiya va tezlik

`jarvis5/research/config.py` — 69 parametr. Presetlar:
`config/research_default.json`, `research_fast.json` (stride 15),
`research_dense.json` (stride 2), `research_60min.json`.

```bash
python -m jarvis5.research --set BASELINE_STRIDE=10 --set EVENT_HORIZON_BARS=60
```

Taxminan **~1 400 M1 bar/sekund**: 6 coin × 365 kun ≈ **35–45 daqiqa**
(stride 5). `research_fast.json` bilan ≈ 15 daqiqa.

## Testlar

```bash
python -m unittest discover -s tests        # 78 test
```

Spec 31 talab qilgan testlar: `test_event_detection`, `test_event_clustering`,
`test_no_lookahead`, `test_liquidity_detection`, `test_smc_detection`,
`test_wyckoff_detection`, `test_rsi_detection`, `test_volume_detection`,
`test_price_action_detection`, `test_combination_detection`, `test_mfe_mae`,
`test_target_detection`, `test_cost_calculation`, `test_risk_calculation`,
`test_oos_split`, `test_report_generation` — hammasi + truncation invariance
va scale invariance.
