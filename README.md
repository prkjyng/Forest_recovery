# 🌲 강원권 산불 피해지 자연복원·인공조림 판단 AI

**팀명**: 강산애(강원도산림愛)
**대회**: 제1회 산림과학 AI 활용 경진대회 (국립산림과학원)  
**팀원**: 충남대학교 산림환경자원학과 재학생 박재용 최석빈 박재윤 · 건국대학교 산림조경학과 졸업생 정민정

---

## 📌 프로젝트 개요

산불 발생 직후 위치 정보(위경도)와 피해 강도(dNBR)를 입력하면  
**자연복원 vs 인공조림** 여부를 AI가 즉시 판단하고 수종을 추천합니다.

### 의사결정 구조
```
🔥 산불 발생
→ 【1단계】 이 모델: "조림이 필요한가?" (즉시 판단)
→ 【2단계】 맞춤형조림지도를 활용한 수종 추천: "어떤 수종을?" (조림 필요 시)
```

### 모델 성능
- 알고리즘: XGBoost(60%) + Random Forest(40%) Soft Voting 앙상블
- CV AUC: **0.8268** ± 0.0189
- Accuracy: 93%
- 학습 데이터: MTfire 강원도 1,472건 (2011~2021)

### 피처 구성 (ndvi 없이 순수 생태 조건만 사용)
| 카테고리 | 피처 | 출처 |
|---|---|---|
| 지형 | elevation, slope, aspect, twi, aspect_northness | SRTM DEM 30m |
| 토양 | drainage_score, weathering_score, frag_score 등 7개 | 산림입지토양도 1:25,000 |
| 기후 | ta_annual, prec_annual, ws_annual, vpd_annual | ERA5-Land (ECMWF) |

### 판단 기준
- 피해 판정: dNBR ≥ 0.10 *(Key & Benson 2006, FIREMON)*
- 산사태 위험: 국립산림과학원 산사태위험지도
- 수종 추천: 맞춤형조림지도 (산림청)

---

## 🗂️ 폴더 구조

```
Forest_recovery/
├── app.py                  ← Streamlit 프로토타입
├── requirements.txt        ← 패키지 목록
├── README.md
└── prototype/
    ├── model_final.pkl         ← 학습된 모델
    ├── mtfire_ndvi_highres.csv ← Sentinel-2/Landsat NDVI
    ├── mtfire_terrain.csv      ← SRTM DEM 지형
    ├── mtfire_soil_imsndo.csv  ← 산림입지토양도 토양
    ├── mtfire_climate.csv      ← ERA5-Land 기후
    ├── mtfire_landslide.csv    ← 산사태위험지도
    └── terrain_lookup.csv      ← 지형 보간용 룩업
```

---

## 🚀 실행 방법

### 1. 환경 설정
```cmd
cd Forest_recovery
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 앱 실행
```cmd
streamlit run app.py
```

브라우저에서 `http://localhost:8501` 접속

---

## 📊 데이터 출처

| 데이터 | 출처 |
|---|---|
| 산불 발생이력 | 산림청 MTfire (2011~2021) |
| NDVI/NBR | GEE Sentinel-2(10m) / Landsat-8(30m) |
| 지형 | SRTM DEM 30m |
| 토양 | 산림청 산림입지토양도 (1:25,000) |
| 기후 | ECMWF ERA5-Land Monthly |
| 산사태 위험 | 국립산림과학원 산사태위험지도 |
| 수종 추천 | 산림청 맞춤형조림지도 |

---

## 📚 참고 문헌

- Key & Benson (2006). FIREMON: Fire Effects Monitoring and Inventory System.
- 최승필·박종선 (2004). 산불피해지역에서 정규산화율지수와 정규식생지수의 비교분석. 한국측량학회지.
- Churchill et al. (2026). Post-fire NDVI-NBR temporal decoupling. ScienceDirect.
- Castellon et al. (2026). Fast NDVI recovery ≠ structural forest recovery. ScienceDirect.
