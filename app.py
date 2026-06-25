"""
app.py - 강원권 산불 피해지 자연복원·인공조림 판단 AI
피처: 지형 + 토양(산림입지도) + 기후(ERA5-Land)
라벨: Sentinel-2/Landsat NDVI 회복률
실행: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import json
import rasterio
from rasterio.transform import rowcol
from pyproj import Transformer
from scipy.spatial import cKDTree
import warnings
warnings.filterwarnings('ignore')

# ── 경로 ──────────────────────────────────────────────────────
BASE           = r"C:\Users\ssoos\Desktop\Forest_recovery\data\prototype"
NDVI_CSV       = os.path.join(BASE, "mtfire_ndvi_highres.csv")
TERRAIN_CSV    = os.path.join(BASE, "mtfire_terrain.csv")
SOIL_CSV       = os.path.join(BASE, "mtfire_soil_imsndo.csv")
CLIMATE_CSV    = os.path.join(BASE, "mtfire_climate.csv")
TERRAIN_LOOKUP = os.path.join(BASE, "terrain_lookup.csv")
MODEL_PATH     = os.path.join(BASE, "model_final.pkl")
GANGWON_CSV    = r"C:\Users\ssoos\Desktop\Forest_recovery\data\gangwon_labeled.csv"
TIF_PATH       = r"C:\Users\ssoos\Desktop\Forest_recovery\data\raw\산사태위험지도\산사태위험지도.tif"

FEATURES = [
    'elevation','slope','aspect','twi','aspect_northness',
    'drainage_score','weathering_score','frag_score',
    'wind_score','altitude_code','soil_type_code','climate_zone',
    'ta_annual','prec_annual','ws_annual','vpd_annual',
]

# ── 유틸 ──────────────────────────────────────────────────────
def classify_severity(dNBR):
    if   dNBR < 0.10: return 'unburned',      '🟢 피해 없음',    '#27ae60'
    elif dNBR < 0.27: return 'low',            '🟡 낮음',         '#f1c40f'
    elif dNBR < 0.44: return 'moderate_low',  '🟠 중간-낮음',    '#e67e22'
    elif dNBR < 0.66: return 'moderate_high', '🔴 중간-높음',    '#e74c3c'
    else:              return 'high',           '⚫ 심각',       '#8e44ad'

def aspect_to_dir(aspect):
    dirs = ['북(N)','북동(NE)','동(E)','남동(SE)','남(S)','남서(SW)','서(W)','북서(NW)']
    return dirs[int((aspect + 22.5) / 45) % 8]

def drainage_label(score):
    return {5:'매우양호',4:'양호',3:'보통',2:'약간불량',1:'불량'}.get(round(score),'보통')

def landslide_color(label):
    return {'매우높음':'#8e44ad','높음':'#e74c3c','보통':'#e67e22',
            '낮음':'#f1c40f','매우낮음':'#27ae60'}.get(label,'#95a5a6')

# ── 리소스 로드 ───────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_resource
def load_terrain_lookup():
    df = pd.read_csv(TERRAIN_LOOKUP)
    # tree: lat, lon 순서
    return df, cKDTree(df[['lat','lon']].values)

@st.cache_resource
def load_soil_lookup():
    soil = pd.read_csv(SOIL_CSV)
    ndvi = pd.read_csv(NDVI_CSV)
    ndvi['fire_id'] = range(len(ndvi))
    def extract_coords(geo_str):
        try:
            geo = json.loads(geo_str)
            return geo['coordinates'][0], geo['coordinates'][1]
        except:
            return None, None
    ndvi['lon'], ndvi['lat'] = zip(*ndvi['.geo'].apply(extract_coords))
    merged = ndvi[['fire_id','lon','lat']].merge(soil, on='fire_id', how='left')
    merged = merged.dropna(subset=['lon','lat'])
    # tree: lat, lon 순서
    return merged, cKDTree(merged[['lat','lon']].values)

@st.cache_resource
def load_climate_lookup():
    climate = pd.read_csv(CLIMATE_CSV)
    climate.rename(columns={'system:index':'idx'}, inplace=True)
    for c in ['ta_annual','prec_annual','ws_annual','vpd_annual']:
        climate[c] = pd.to_numeric(climate[c], errors='coerce')
        climate.loc[climate[c] < -1000, c] = np.nan
    ndvi = pd.read_csv(NDVI_CSV)
    ndvi['fire_id'] = range(len(ndvi))
    def extract_coords(geo_str):
        try:
            geo = json.loads(geo_str)
            return geo['coordinates'][0], geo['coordinates'][1]
        except:
            return None, None
    ndvi['lon'], ndvi['lat'] = zip(*ndvi['.geo'].apply(extract_coords))
    climate['fire_id'] = range(len(climate))
    merged = ndvi[['fire_id','lon','lat']].merge(
        climate[['fire_id','ta_annual','prec_annual','ws_annual','vpd_annual']],
        on='fire_id', how='left')
    merged = merged.dropna(subset=['lon','lat'])
    # tree: lat, lon 순서
    return merged, cKDTree(merged[['lat','lon']].values)

@st.cache_resource
def load_species_lookup():
    try:
        df = pd.read_csv(GANGWON_CSV, encoding='utf-8-sig')
        df = df.dropna(subset=['lat','lon','추천수종_적지'])
        return df, cKDTree(df[['lat','lon']].values)
    except:
        return None, None

# ── 데이터 조회 함수 (모두 lat, lon 순서로 통일) ─────────────
def get_terrain(lat, lon):
    try:
        df_t, tree = load_terrain_lookup()
        dist, idx = tree.query([[lat, lon]], k=3)
        w = 1/(dist[0]+1e-10); w /= w.sum()
        row = df_t.iloc[idx[0]]
        return tuple(float(np.average(row[c], weights=w))
                     for c in ['elevation','slope','aspect','twi','aspect_northness'])
    except:
        return 300.0, 15.0, 180.0, 2.0, 90.0

def get_soil(lat, lon):
    SOIL_COLS = ['drainage_score','weathering_score','frag_score',
                 'wind_score','altitude_code','soil_type_code','climate_zone']
    try:
        df_s, tree = load_soil_lookup()
        dist, idx = tree.query([[lat, lon]], k=3)
        w = 1/(dist[0]+1e-10); w /= w.sum()
        row = df_s.iloc[idx[0]]
        return {c: float(np.average(row[c], weights=w)) for c in SOIL_COLS}
    except:
        return {c: 3.0 for c in SOIL_COLS}

def get_climate(lat, lon):
    CLIM_COLS = ['ta_annual','prec_annual','ws_annual','vpd_annual']
    try:
        df_c, tree = load_climate_lookup()
        dist, idx = tree.query([[lat, lon]], k=3)
        w = 1/(dist[0]+1e-10); w /= w.sum()
        row = df_c.iloc[idx[0]]
        return {c: float(np.average(row[c], weights=w)) for c in CLIM_COLS}
    except:
        return {'ta_annual':10.0,'prec_annual':100.0,'ws_annual':2.0,'vpd_annual':5.0}

def get_landslide(lat, lon):
    try:
        t = Transformer.from_crs('EPSG:4326','EPSG:5181', always_xy=True)
        x, y = t.transform(lon, lat)
        with rasterio.open(TIF_PATH) as src:
            r, c = rowcol(src.transform, x, y)
            if 0 <= r < src.height and 0 <= c < src.width:
                val = int(src.read(1)[r, c])
                if src.nodata and val == int(src.nodata):
                    return -1, '데이터없음'
                return val, {1:'매우높음',2:'높음',3:'보통',4:'낮음',5:'매우낮음'}.get(val,'알수없음')
        return -1, '데이터없음'
    except:
        return -1, '데이터없음'

def get_species(lat, lon):
    try:
        df_sp, tree = load_species_lookup()
        if df_sp is None: return None, None
        dist, idx = tree.query([[lat, lon]], k=1)
        row = df_sp.iloc[idx[0]]
        return row.get('추천수종_적지',''), row.get('추천수종_가능','')
    except:
        return None, None

# ══════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════
st.set_page_config(page_title="산불 피해지 복원 판단 AI", page_icon="🌲", layout="wide")

st.title("🌲 강원권 산불 피해지 자연복원·인공조림 판단 AI")
st.markdown("""
<div style='background:#1a1a2e;border-radius:8px;padding:12px;margin-bottom:8px'>
<b>🔥 산불 발생</b> →
<b style='color:#f39c12'>【1단계】 이 모델</b>
<span style='color:#aaa'>: 자연복원 vs 인공조림 판단 (CV AUC 0.8268)</span> →
<b style='color:#27ae60'>【2단계】 맞춤형조림지도</b>
<span style='color:#aaa'>: 수종 추천</span>
</div>
<div style='color:#888;font-size:12px'>
피처: 지형(SRTM) + 토양(산림입지토양도) + 기후(ERA5-Land) | 라벨: Sentinel-2/Landsat NDVI 3년 회복률
</div>
""", unsafe_allow_html=True)
st.divider()

artifact = load_model()
model    = artifact['model']

# ── STEP 1: 위치 ──────────────────────────────────────────────
st.subheader("📍 STEP 1 · 산불 발생 위치")
col1, col2 = st.columns(2)
with col1:
    lat = st.number_input("위도", 37.0, 38.7, 37.85, 0.001, format="%.5f")
with col2:
    lon = st.number_input("경도", 127.0, 129.5, 128.56, 0.001, format="%.5f")

elevation, slope, aspect, twi, aspect_northness = get_terrain(lat, lon)
soil_vals    = get_soil(lat, lon)
climate_vals = get_climate(lat, lon)
ls_val, ls_label = get_landslide(lat, lon)
lc = landslide_color(ls_label)

with st.expander("🗺️ 자동 추출된 입지·기후 정보"):
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("고도", f"{elevation:.0f}m")
    c2.metric("경사도", f"{slope:.1f}°")
    c3.metric("사면방위", aspect_to_dir(aspect))
    c4.metric("배수등급", drainage_label(soil_vals['drainage_score']))
    c1.metric("연평균기온", f"{climate_vals['ta_annual']:.1f}℃")
    c2.metric("연강수량", f"{climate_vals['prec_annual']:.0f}mm")
    c3.metric("VPD", f"{climate_vals['vpd_annual']:.2f}")
    c4.markdown(
        f"**산사태 위험**  \n"
        f"<span style='color:{lc};font-size:16px;font-weight:bold'>{ls_label}</span>",
        unsafe_allow_html=True)
    st.caption("출처: SRTM DEM | 산림입지토양도(1:25,000) | ERA5-Land | 산사태위험지도")

st.divider()

# ── STEP 2: 피해 강도 ─────────────────────────────────────────
st.subheader("🔥 STEP 2 · 산불 피해 강도")
dNBR = st.slider("dNBR (산불 피해 강도)", -0.1, 0.8, 0.15, 0.01,
                 help="dNBR = NBR_pre - NBR_post (Sentinel-2/Landsat 산출)")
sev_code, sev_label, sev_color = classify_severity(dNBR)
st.markdown(
    f"**피해 강도:** <span style='color:{sev_color};font-weight:bold'>{sev_label}</span> "
    f"<span style='color:#888;font-size:11px'>(Key & Benson 2006, FIREMON)</span>",
    unsafe_allow_html=True)

st.divider()

# ── STEP 3: 판단 결과 ─────────────────────────────────────────
st.subheader("🔍 STEP 3 · 【1단계】 복원 방법 판단")

if sev_code == 'unburned':
    st.success("## ✅ 복원 조치 불필요")
    st.markdown(f"dNBR **{dNBR:.2f}** < 0.10 *(Key & Benson 2006)*")
    st.stop()

# 입력 벡터 구성
X_in = pd.DataFrame([[
    elevation, slope, aspect, twi, aspect_northness,
    soil_vals['drainage_score'], soil_vals['weathering_score'],
    soil_vals['frag_score'], soil_vals['wind_score'],
    soil_vals['altitude_code'], soil_vals['soil_type_code'], soil_vals['climate_zone'],
    climate_vals['ta_annual'], climate_vals['prec_annual'],
    climate_vals['ws_annual'], climate_vals['vpd_annual'],
]], columns=FEATURES)

prob     = model.predict_proba(X_in)[0]
prob_자연 = prob[0]
prob_조림  = prob[1]
pred     = int(prob_조림 >= 0.5)

col_main, col_side = st.columns([2.5, 1])

with col_side:
    st.metric("🌿 자연복원 확률", f"{prob_자연*100:.1f}%")
    st.metric("🌱 인공조림 확률", f"{prob_조림*100:.1f}%")
    st.progress(float(prob_조림))
    if max(prob_자연, prob_조림) < 0.65:
        st.warning("확률 낮음\n현장 확인 권장")
    st.markdown("---")
    st.markdown(
        f"**⛰️ 산사태 위험**  \n"
        f"<span style='color:{lc};font-size:16px;font-weight:bold'>{ls_label}</span>",
        unsafe_allow_html=True)
    if ls_label in ['매우높음','높음']:
        st.error("사방공사 우선 검토")
    st.caption("*CV AUC 0.8268 · 1,472건 학습*")

with col_main:
    st.markdown(
        f"<div style='background:{sev_color}22;border-left:4px solid {sev_color};"
        f"padding:10px;border-radius:4px;margin-bottom:12px'>"
        f"<b>피해 강도:</b> {sev_label} &nbsp;|&nbsp; dNBR {dNBR:.2f} &nbsp;|&nbsp; "
        f"기온 {climate_vals['ta_annual']:.1f}℃ &nbsp;|&nbsp; "
        f"강수 {climate_vals['prec_annual']:.0f}mm &nbsp;|&nbsp; "
        f"VPD {climate_vals['vpd_annual']:.2f}"
        f"</div>", unsafe_allow_html=True)

    if pred == 0:
        st.success("## ✅ 자연복원 권고")
        st.markdown(f"""
        **판단 근거:**
        - 해당 지점 토양·지형·기후 조건에서 과거 산불 피해지가 3년 내 자연회복
        - 강수량 **{climate_vals['prec_annual']:.0f}mm** / VPD **{climate_vals['vpd_annual']:.2f}** → 수분 조건 양호
        - 배수등급 **{drainage_label(soil_vals['drainage_score'])}** → 식생 활착 유리

        **권고사항:**
        - 자연 천이 유도 · 불필요한 조림 비용 절감
        - *(조림 비용: 약 983만원/ha, 산림청 2022)*
        - 3~5년 주기 위성 NDVI 모니터링 지속
        """)
        st.info("**맞춤형조림지도 2단계 불필요** — 자연복원으로 충분합니다.")

    else:
        st.error("## ⚠️ 인공조림 권고")
        st.markdown(f"""
        **판단 근거:**
        - 해당 지점 토양·지형·기후 조건에서 과거 산불 피해지가 3년 내 자연회복 어려움
        - 강수량 **{climate_vals['prec_annual']:.0f}mm** / VPD **{climate_vals['vpd_annual']:.2f}** → 수분 스트레스
        - 피해 강도 **{sev_label}** → 자연 종자은행·뿌리 손상 우려
        """)

        if ls_label in ['매우높음','높음']:
            st.warning(f"⚠️ 산사태 위험 **{ls_label}** — 조림 전 사방공사 우선 시행 권고")

        st.divider()
        st.subheader("🌳 【2단계】 맞춤형조림지도 — 수종 추천")

        species_적지, species_가능 = get_species(lat, lon)
        if species_적지 and str(species_적지) not in ['','nan','None']:
            st.success(f"**적지 수종**: {species_적지}")
            if species_가능 and str(species_가능) not in ['','nan','None']:
                st.info(f"**가능 수종**: {species_가능}")
            st.caption("출처: 맞춤형조림지도(산림청)")
        else:
            지역 = "온대북부" if lat>=38.0 else "온대중부" if lat>=37.5 else "온대남부"
            수종  = "잣나무·자작나무·전나무" if lat>=38.0 else "소나무·낙엽송·신갈나무" if lat>=37.5 else "소나무·상수리나무·굴참나무"
            st.info(f"**기후대**: {지역} | **권장 수종**: {수종}")
            st.caption("맞춤형조림지도 조회 불가 → 위도 기반 기후대 적용")

        st.markdown("**최적 조림 시기**: 봄 3월 하순-4월 | 가을 10월-11월 상순")

# ── 하단 ──────────────────────────────────────────────────────
st.divider()
with st.expander("ℹ️ 모델 정보 및 판단 기준"):
    st.markdown("""
    **의사결정 구조**  
    【1단계】 이 모델: "조림이 필요한가?" → 산불 직후 즉시 판단  
    【2단계】 맞춤형조림지도: "어떤 수종을?" → 조림 필요 확정 시

    **모델**  
    - 피처: 지형(SRTM DEM) + 토양(산림입지토양도 1:25,000) + 기후(ERA5-Land)  
    - 라벨: Sentinel-2(10m)/Landsat-8(30m) NDVI 3년 회복률 < 1.0 → 인공조림  
    - 알고리즘: XGBoost(60%) + Random Forest(40%) Soft Voting 앙상블  
    - 성능: **CV AUC 0.8268** · Accuracy 93% · 학습 1,472건  
    - SHAP 1위: 강수량(prec_annual) → 2위: VPD → 3위: 풍속

    **판단 기준**  
    - 피해 판정: dNBR ≥ 0.10 *(Key & Benson 2006, FIREMON)*  
    - 산사태: 국립산림과학원 산사태위험지도  
    - 수종: 맞춤형조림지도(산림청)
    """)

st.caption("제1회 산림과학 AI 활용 경진대회 | 강산애(강원도산림愛)")
