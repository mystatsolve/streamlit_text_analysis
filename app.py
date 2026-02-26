# -*- coding: utf-8 -*-
"""
한국어 텍스트 분석 앱 @AI개발자 김우진이 만듬, AI개발 및 데이터 분석 문의: mystatsolve@gmail.com
- CSV 업로드 (sentence/abstract + year 컬럼)
- Kiwi 형태소 분석기로 명사 추출
- 불용어 사용자 설정
- 빈도 분석표 + 워드 클라우드
- 결과 CSV 다운로드
"""

import io
import os
import warnings
warnings.filterwarnings("ignore")

# tensorflow(umap 경유)의 protobuf 버전 충돌 방지
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from collections import Counter
from pathlib import Path
from wordcloud import WordCloud
import streamlit as st
import streamlit.components.v1 as components
from kiwipiepy import Kiwi
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from itertools import combinations
import networkx as nx

# ── 페이지 설정 ────────────────────────────────────────────
st.set_page_config(
    page_title="텍스트 빈도 분석",
    page_icon="📊",
    layout="wide",
)

# ── CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
  .main-title {font-size:2rem; font-weight:800; margin-bottom:.2rem;}
  .sub-title  {color:#64748b; font-size:.95rem; margin-bottom:1.5rem;}
  .step-label {
    background:#3b82f6; color:#fff;
    border-radius:50%; width:24px; height:24px;
    display:inline-flex; align-items:center; justify-content:center;
    font-size:.78rem; font-weight:700; margin-right:6px;
  }
  .section-title {font-size:1.05rem; font-weight:700; margin-bottom:.6rem;}
  div[data-testid="stSidebar"] .stButton > button {width:100%;}
</style>
""", unsafe_allow_html=True)

# ── 한국어 폰트 설정 (워드클라우드 & matplotlib) ───────────
@st.cache_resource
def get_korean_font():
    """시스템에서 한글 지원 폰트를 탐색해 경로 반환"""
    candidates = [
        "C:/Windows/Fonts/malgun.ttf",          # 맑은 고딕 (Windows)
        "C:/Windows/Fonts/NanumGothic.ttf",
        "C:/Windows/Fonts/gulim.ttc",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",       # Linux (fonts-nanum)
        "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf",  # Linux 대체
        "/usr/share/fonts/truetype/nanum/NanumMyeongjo.ttf",     # Linux 대체2
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",            # macOS
    ]
    for p in candidates:
        if Path(p).exists():
            return p
    # matplotlib 폰트 리스트에서 한글 지원 폰트 탐색
    for f in fm.findSystemFonts():
        try:
            prop = fm.FontProperties(fname=f)
            if any(k in prop.get_name() for k in ["Gothic", "고딕", "Nanum", "Malgun", "나눔"]):
                return f
        except Exception:
            pass
    return None

FONT_PATH = get_korean_font()
FONT_NAME = "sans-serif"

if FONT_PATH:
    fm.fontManager.addfont(FONT_PATH)
    _prop = fm.FontProperties(fname=FONT_PATH)
    FONT_NAME = _prop.get_name()
    plt.rcParams["font.family"] = FONT_NAME
    plt.rcParams["font.sans-serif"] = [FONT_NAME] + plt.rcParams.get("font.sans-serif", [])
else:
    plt.rcParams["font.family"] = "sans-serif"

plt.rcParams["axes.unicode_minus"] = False

# ── Kiwi 초기화 (캐시) ───────────────────────────────────
@st.cache_resource(show_spinner="형태소 분석기 로딩 중…")
def load_kiwi():
    return Kiwi()

# ── 기본 불용어 ──────────────────────────────────────────
DEFAULT_STOPWORDS = [
    # 기존 코드 불용어
    "것", "등", "수", "년", "월", "일", "바", "점", "중", "내", "후", "전",
    "간", "상", "때", "곳", "말", "분", "명", "개", "회", "권", "편",
    "부", "면", "장", "절", "항", "호", "차", "건", "대", "위", "표",
    "그림", "참고", "예", "단", "쪽", "페이지", "학술", "논문",
    # 추가 일반 불용어
    "이", "가", "은", "는", "을", "를", "의", "에", "와", "과",
    "도", "로", "으로", "에서", "에게", "보다", "이다", "있다", "없다",
    "하다", "되다", "않다", "그리고", "하지만", "그러나", "또한",
    "그래서", "따라서", "그런데", "본", "각", "여러", "모든",
]

# ── 명사 추출 함수 ───────────────────────────────────────
NOUN_TAGS = {"NNG", "NNP"}  # 일반명사, 고유명사

def extract_nouns(kiwi: Kiwi, texts: list[str], stopwords: set, min_len: int = 2) -> list[str]:
    nouns = []
    for text in texts:
        try:
            result = kiwi.analyze(text)
            for token, tag, _, _ in result[0][0]:
                if tag in NOUN_TAGS and len(token) >= min_len and token not in stopwords:
                    nouns.append(token)
        except Exception:
            pass
    return nouns

# ── TF-IDF용: 문서별 명사 문자열 변환 ───────────────────
def texts_to_noun_docs(kiwi: Kiwi, texts: list[str], stopwords: set, min_len: int = 2) -> list[str]:
    """각 문서를 '명사 명사 명사…' 형태의 문자열로 변환 (TfidfVectorizer 입력용)"""
    docs = []
    for text in texts:
        try:
            result = kiwi.analyze(str(text))
            nouns = [tok for tok, tag, _, _ in result[0][0]
                     if tag in NOUN_TAGS and len(tok) >= min_len and tok not in stopwords]
            docs.append(" ".join(nouns))
        except Exception:
            docs.append("")
    return docs

# ── 워드클라우드 생성 ────────────────────────────────────
def make_wordcloud(freq_dict: dict, font_path: str, max_words: int, colormap: str):
    wc = WordCloud(
        font_path=font_path if font_path else None,
        background_color="white",
        width=900,
        height=500,
        max_words=max_words,
        colormap=colormap,
        prefer_horizontal=0.9,
    )
    wc.generate_from_frequencies(freq_dict)
    return wc

# ── PNG 저장 헬퍼 (전역) ────────────────────────────────
def save_fig_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="PNG", dpi=150, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    return buf.getvalue()

# ── 메인 UI ──────────────────────────────────────────────
st.markdown('<p class="main-title">📊 텍스트 빈도 분석 도구</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">CSV 업로드 → 불용어 설정 → 빈도 분석 & 워드 클라우드</p>', unsafe_allow_html=True)
st.markdown(
    '<p style="font-size:0.85rem; color:#475569; margin-top:-0.8rem; margin-bottom:1rem;">'
    'AI개발자 김우진 제작 &nbsp;|&nbsp; AI개발 및 데이터 분석 의뢰: '
    '<a href="mailto:mystatsolve@gmail.com" style="color:#3b82f6;">mystatsolve@gmail.com</a>'
    '</p>',
    unsafe_allow_html=True,
)

# ════════════════════════════════════════════════════════
# STEP 1: CSV 업로드
# ════════════════════════════════════════════════════════
st.markdown('<span class="step-label">1</span><span class="section-title">CSV 파일 업로드</span>', unsafe_allow_html=True)

uploaded = st.file_uploader(
    "sentence(또는 abstract) · year 컬럼이 포함된 CSV 파일을 선택하세요",
    type=["csv"],
    label_visibility="collapsed",
)

if uploaded is None:
    st.info("📂 CSV 파일을 업로드하면 분석을 시작할 수 있습니다.")
    st.stop()

# CSV 파싱
@st.cache_data(show_spinner=False)
def load_csv(file_bytes: bytes, file_name: str) -> pd.DataFrame:
    for enc in ["utf-8-sig", "utf-8", "cp949", "euc-kr"]:
        try:
            return pd.read_csv(io.BytesIO(file_bytes), encoding=enc)
        except Exception:
            pass
    return pd.DataFrame()

raw_bytes = uploaded.read()
df_raw = load_csv(raw_bytes, uploaded.name)

if df_raw.empty:
    st.error("CSV 파일을 읽을 수 없습니다. 인코딩을 확인해주세요.")
    st.stop()

st.success(f"✅ **{uploaded.name}** 로드 완료 — {len(df_raw):,}행 · {len(df_raw.columns)}열")

# 컬럼 자동 감지
TEXT_HINTS = ["sentence", "abstract", "text", "텍스트", "문장", "내용", "초록"]
YEAR_HINTS = ["year", "연도", "년도", "년", "Year"]

auto_text = next((c for c in df_raw.columns if c.lower() in [h.lower() for h in TEXT_HINTS]), df_raw.columns[0])
auto_year = next((c for c in df_raw.columns if c.lower() in [h.lower() for h in YEAR_HINTS]), df_raw.columns[-1])

c1, c2 = st.columns(2)
with c1:
    text_col = st.selectbox("텍스트 컬럼", df_raw.columns.tolist(), index=df_raw.columns.tolist().index(auto_text))
with c2:
    year_col = st.selectbox("연도 컬럼",  df_raw.columns.tolist(), index=df_raw.columns.tolist().index(auto_year))

# ════════════════════════════════════════════════════════
# STEP 2: 분석 설정 (사이드바)
# ════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ 분석 설정")

    # 연도 필터
    st.markdown("**연도 필터**")
    all_years = sorted(df_raw[year_col].dropna().astype(str).unique().tolist())
    sel_years = st.multiselect("분석할 연도 (비우면 전체)", all_years, default=[])

    st.divider()

    # 빈도 설정
    st.markdown("**분석 옵션**")
    min_freq = st.slider("최소 빈도수",    1, 20, 2)
    min_len  = st.slider("최소 단어 길이", 1,  5, 2)
    top_n    = st.slider("상위 단어 수",  10, 300, 100)

    st.divider()

    # 불용어 설정
    st.markdown("**불용어 설정**")
    sw_text = st.text_area(
        "불용어 목록 (한 줄에 하나 또는 쉼표 구분)",
        value="\n".join(DEFAULT_STOPWORDS),
        height=220,
    )
    stopwords_set = {
        w.strip()
        for line in sw_text.splitlines()
        for w in line.split(",")
        if w.strip()
    }
    st.caption(f"등록된 불용어: {len(stopwords_set)}개")

    st.divider()

    # 워드클라우드 설정
    st.markdown("**워드 클라우드 설정**")
    wc_max   = st.slider("워드 클라우드 단어 수", 20, 200, 80)
    wc_cmap  = st.selectbox(
        "색상 테마",
        ["Blues", "viridis", "plasma", "Set2", "tab10", "RdYlBu", "coolwarm"],
    )

    st.divider()
    run_btn = st.button("🔍 분석 시작", type="primary", use_container_width=True)

# ════════════════════════════════════════════════════════
# STEP 3: 분석 실행
# ════════════════════════════════════════════════════════
if not run_btn:
    st.info("👈 왼쪽 사이드바에서 설정을 완료하고 **분석 시작** 버튼을 눌러주세요.")
    st.stop()

# 연도 필터 적용
df_work = df_raw.copy()
df_work[year_col] = df_work[year_col].astype(str).str.strip()
if sel_years:
    df_work = df_work[df_work[year_col].isin(sel_years)]

texts = df_work[text_col].dropna().astype(str).tolist()

if not texts:
    st.error("분석할 텍스트 데이터가 없습니다. 연도 필터 또는 컬럼 설정을 확인해주세요.")
    st.stop()

# Kiwi 로드 & 명사 추출
kiwi = load_kiwi()

with st.spinner(f"🔎 {len(texts):,}개 문서에서 명사 추출 중… (잠시 기다려주세요)"):
    all_nouns = extract_nouns(kiwi, texts, stopwords_set, min_len)

if not all_nouns:
    st.warning("추출된 명사가 없습니다. 불용어 목록이나 최소 단어 길이를 조정해보세요.")
    st.stop()

# 빈도 계산
counter = Counter(all_nouns)
filtered = [(w, f) for w, f in counter.most_common() if f >= min_freq][:top_n]

if not filtered:
    st.warning(f"최소 빈도 {min_freq} 이상인 단어가 없습니다. 최소 빈도를 낮춰보세요.")
    st.stop()

total_freq = sum(f for _, f in filtered)
freq_df = pd.DataFrame(
    [{"순위": i+1, "단어": w, "빈도": f, "비율(%)": round(f / total_freq * 100, 2)}
     for i, (w, f) in enumerate(filtered)]
)

# ── 통계 요약 ────────────────────────────────────────────
st.markdown("---")
m1, m2, m3, m4 = st.columns(4)
m1.metric("분석 문서 수",  f"{len(texts):,}")
m2.metric("전체 명사 수",  f"{len(all_nouns):,}")
m3.metric("고유 단어 수",  f"{len(counter):,}")
m4.metric("분석 연도 수",  len(sel_years) if sel_years else len(all_years))

# ════════════════════════════════════════════════════════
# STEP 4: 결과 탭
# ════════════════════════════════════════════════════════
tab_freq, tab_wc, tab_yearly, tab_tfidf, tab_cooc = st.tabs([
    "📋 빈도 분석표", "☁️ 워드 클라우드", "📅 연도별 분석", "📈 TF-IDF 분석", "🕸️ 동시출현 네트워크",
])

# ── 빈도 분석표 ──────────────────────────────────────────
with tab_freq:
    col_left, col_right = st.columns([3, 1])
    with col_left:
        st.markdown(f"**상위 {len(freq_df)}개 단어**")
    with col_right:
        csv_bytes = freq_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button(
            label="⬇ CSV 다운로드",
            data=csv_bytes,
            file_name="빈도분석_결과.csv",
            mime="text/csv",
        )

    # 막대 그래프 (상위 30개)
    top30 = freq_df.head(30)
    fig, ax = plt.subplots(figsize=(10, max(4, len(top30) * 0.32)))
    bars = ax.barh(top30["단어"][::-1], top30["빈도"][::-1], color="#3b82f6", alpha=0.85)
    ax.bar_label(bars, padding=3, fontsize=8)
    ax.set_xlabel("빈도")
    ax.set_title(f"상위 {len(top30)}개 빈출 단어", fontsize=13, fontweight="bold", pad=10)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # 데이터 테이블
    st.dataframe(
        freq_df,
        column_config={
            "순위": st.column_config.NumberColumn(width="small"),
            "빈도": st.column_config.NumberColumn(format="%d"),
            "비율(%)": st.column_config.NumberColumn(format="%.2f"),
        },
        use_container_width=True,
        hide_index=True,
        height=400,
    )

# ── 워드 클라우드 ────────────────────────────────────────
with tab_wc:
    if FONT_PATH is None:
        st.warning("한글 폰트를 찾지 못했습니다. 워드 클라우드의 한글이 깨질 수 있습니다.")

    with st.spinner("워드 클라우드 생성 중…"):
        freq_dict = {row["단어"]: row["빈도"] for _, row in freq_df.head(wc_max).iterrows()}
        try:
            wc = make_wordcloud(freq_dict, FONT_PATH, wc_max, wc_cmap)
            fig_wc, ax_wc = plt.subplots(figsize=(12, 6))
            ax_wc.imshow(wc, interpolation="bilinear")
            ax_wc.axis("off")
            plt.tight_layout(pad=0)
            st.pyplot(fig_wc)
            plt.close(fig_wc)

            # PNG 다운로드
            wc_buf = io.BytesIO()
            wc.to_image().save(wc_buf, format="PNG")
            st.download_button(
                label="🖼 워드 클라우드 PNG 저장",
                data=wc_buf.getvalue(),
                file_name="워드클라우드.png",
                mime="image/png",
            )
        except Exception as e:
            st.error(f"워드 클라우드 생성 오류: {e}")

# ════════════════════════════════════════════════════════
# STEP 5: 연도별 분석 탭
# ════════════════════════════════════════════════════════
with tab_yearly:

    # 분석 대상 연도 목록 (필터 적용된 데이터 기준)
    yearly_targets = sorted(df_work[year_col].dropna().astype(str).str.strip().unique().tolist())

    if len(yearly_targets) == 0:
        st.warning("분석 가능한 연도 데이터가 없습니다.")

    # ── 연도별 명사 추출 & 빈도 계산 ─────────────────────
    with st.spinner(f"📅 {len(yearly_targets)}개 연도별 분석 중…"):
        yearly_results = {}   # {year: Counter}
        all_yearly_rows = []  # 상위30 CSV용

        for yr in yearly_targets:
            yr_texts = (
                df_work[df_work[year_col].astype(str).str.strip() == yr][text_col]
                .dropna().astype(str).tolist()
            )
            yr_nouns = extract_nouns(kiwi, yr_texts, stopwords_set, min_len)
            yr_counter = Counter(yr_nouns)
            yearly_results[yr] = yr_counter

            total_yr = len(yr_nouns)
            for rank, (noun, freq) in enumerate(yr_counter.most_common(30), 1):
                ratio = round(freq / total_yr * 100, 2) if total_yr > 0 else 0
                all_yearly_rows.append({
                    "연도": yr, "순위": rank,
                    "단어": noun, "빈도": freq, "비율(%)": ratio,
                })

    # ── 다운로드 버튼 ────────────────────────────────────
    dl1, dl2 = st.columns(2)

    # 연도별 상위 30 CSV
    yearly_df = pd.DataFrame(all_yearly_rows)
    with dl1:
        st.download_button(
            label="⬇ 연도별 상위 30 CSV",
            data=yearly_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
            file_name="빈도분석_연도별_상위30.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # 피벗 테이블 CSV (단어 × 연도)
    all_pivot_words = set()
    for yr, cnt in yearly_results.items():
        for w, _ in cnt.most_common(50):
            all_pivot_words.add(w)

    pivot_rows = []
    for word in all_pivot_words:
        row = {"단어": word}
        for yr in yearly_targets:
            row[str(yr)] = yearly_results[yr].get(word, 0)
        pivot_rows.append(row)

    pivot_df = pd.DataFrame(pivot_rows)
    pivot_df["총합"] = pivot_df[[str(y) for y in yearly_targets]].sum(axis=1)
    pivot_df = pivot_df.sort_values("총합", ascending=False).drop("총합", axis=1)

    with dl2:
        st.download_button(
            label="⬇ 연도별 피벗 테이블 CSV",
            data=pivot_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
            file_name="빈도분석_연도별_피벗.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.divider()

    # ── 연도별 상위 15 단어 막대 그래프 ─────────────────
    st.markdown("#### 📊 연도별 상위 15개 빈출 단어")

    n_cols = 2
    yr_chunks = [yearly_targets[i:i+n_cols] for i in range(0, len(yearly_targets), n_cols)]

    for chunk in yr_chunks:
        cols = st.columns(n_cols)
        for ci, yr in enumerate(chunk):
            yr_counter = yearly_results[yr]
            top15 = yr_counter.most_common(15)
            if not top15:
                with cols[ci]:
                    st.caption(f"{yr}년: 데이터 없음")
                continue

            words  = [w for w, _ in top15][::-1]
            freqs  = [f for _, f in top15][::-1]
            max_f  = max(freqs)

            fig_yr, ax_yr = plt.subplots(figsize=(5, max(3, len(words) * 0.35)))
            bars_yr = ax_yr.barh(words, freqs, color="#3b82f6", alpha=0.82)
            ax_yr.bar_label(bars_yr, padding=3, fontsize=7)
            ax_yr.set_title(f"{yr}년 상위 {len(words)}개", fontsize=11, fontweight="bold")
            ax_yr.set_xlim(0, max_f * 1.18)
            ax_yr.spines[["top", "right"]].set_visible(False)
            ax_yr.tick_params(labelsize=8)
            plt.tight_layout()

            with cols[ci]:
                st.pyplot(fig_yr)
            plt.close(fig_yr)

    st.divider()

    # ── 연도별 워드 클라우드 그리드 ──────────────────────
    st.markdown("#### ☁️ 연도별 워드 클라우드")

    if FONT_PATH is None:
        st.warning("한글 폰트를 찾지 못해 워드 클라우드 한글이 깨질 수 있습니다.")

    wc_cols = st.columns(n_cols)
    for ci, yr in enumerate(yearly_targets):
        yr_freq = dict(yearly_results[yr].most_common(wc_max))
        if not yr_freq:
            continue
        try:
            wc_yr = WordCloud(
                font_path=FONT_PATH if FONT_PATH else None,
                background_color="white",
                width=700, height=420,
                max_words=wc_max,
                colormap=wc_cmap,
                prefer_horizontal=0.9,
            ).generate_from_frequencies(yr_freq)

            fig_wc_yr, ax_wc_yr = plt.subplots(figsize=(6, 3.5))
            ax_wc_yr.imshow(wc_yr, interpolation="bilinear")
            ax_wc_yr.axis("off")
            ax_wc_yr.set_title(f"{yr}년 주요 키워드", fontsize=11, fontweight="bold", pad=6)
            plt.tight_layout(pad=0.5)

            with wc_cols[ci % n_cols]:
                st.pyplot(fig_wc_yr)

                # 연도별 PNG 다운로드
                buf_yr = io.BytesIO()
                wc_yr.to_image().save(buf_yr, format="PNG")
                st.download_button(
                    label=f"🖼 {yr}년 PNG 저장",
                    data=buf_yr.getvalue(),
                    file_name=f"워드클라우드_{yr}.png",
                    mime="image/png",
                    use_container_width=True,
                )
            plt.close(fig_wc_yr)
        except Exception as e:
            with wc_cols[ci % n_cols]:
                st.error(f"{yr}년 워드 클라우드 오류: {e}")

    st.divider()

    # ── 전체 통합 워드 클라우드 ───────────────────────────
    st.markdown("#### ☁️ 전체 기간 통합 워드 클라우드")

    all_combined = Counter()
    for cnt in yearly_results.values():
        all_combined.update(cnt)

    if all_combined:
        try:
            wc_all = WordCloud(
                font_path=FONT_PATH if FONT_PATH else None,
                background_color="white",
                width=1100, height=600,
                max_words=100,
                colormap=wc_cmap,
                prefer_horizontal=0.9,
            ).generate_from_frequencies(dict(all_combined.most_common(100)))

            fig_all, ax_all = plt.subplots(figsize=(13, 6))
            ax_all.imshow(wc_all, interpolation="bilinear")
            ax_all.axis("off")
            yr_range = f"{min(yearly_targets)}~{max(yearly_targets)}"
            ax_all.set_title(f"{yr_range}년 전체 주요 키워드", fontsize=14, fontweight="bold", pad=8)
            plt.tight_layout(pad=0.5)
            st.pyplot(fig_all)
            plt.close(fig_all)

            buf_all = io.BytesIO()
            wc_all.to_image().save(buf_all, format="PNG")
            st.download_button(
                label="🖼 전체 통합 워드 클라우드 PNG 저장",
                data=buf_all.getvalue(),
                file_name="워드클라우드_전체.png",
                mime="image/png",
            )
        except Exception as e:
            st.error(f"전체 워드 클라우드 오류: {e}")

# ════════════════════════════════════════════════════════
# STEP 6: TF-IDF 분석 탭
# ════════════════════════════════════════════════════════
with tab_tfidf:

    # ── 문서별 명사 문자열 변환 ───────────────────────────
    with st.spinner(f"📈 TF-IDF 분석을 위한 명사 추출 중…"):
        noun_docs = texts_to_noun_docs(kiwi, texts, stopwords_set, min_len)

    valid_idx  = [i for i, d in enumerate(noun_docs) if d.strip()]
    valid_docs = [noun_docs[i] for i in valid_idx]

    if len(valid_docs) < 2:
        st.warning("TF-IDF 분석을 위한 유효 문서가 2개 미만입니다. 데이터를 확인해주세요.")

    # ══════════════════════════════════════════════════
    # 섹션 A: 전체 TF-IDF
    # ══════════════════════════════════════════════════
    st.markdown("#### 📈 전체 TF-IDF 분석")

    with st.spinner("전체 TF-IDF 계산 중…"):
        vec_all = TfidfVectorizer(max_features=500)
        mat_all = vec_all.fit_transform(valid_docs)
        feat_all   = vec_all.get_feature_names_out()
        means_all  = mat_all.mean(axis=0).A1

    tfidf_all_df = (
        pd.DataFrame({"단어": feat_all, "TF-IDF 평균": means_all.round(6)})
        .sort_values("TF-IDF 평균", ascending=False)
        .reset_index(drop=True)
    )
    tfidf_all_df.insert(0, "순위", range(1, len(tfidf_all_df) + 1))
    tfidf_top = tfidf_all_df.head(top_n)

    # 다운로드
    dl_tfidf_all, _ = st.columns([1, 3])
    with dl_tfidf_all:
        st.download_button(
            label="⬇ 전체 TF-IDF CSV",
            data=tfidf_all_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
            file_name="tfidf_분석결과.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # 막대 그래프 (상위 30)
    t30 = tfidf_top.head(30)
    fig_t, ax_t = plt.subplots(figsize=(10, max(4, len(t30) * 0.32)))
    bars_t = ax_t.barh(t30["단어"][::-1], t30["TF-IDF 평균"][::-1], color="#8b5cf6", alpha=0.85)
    ax_t.bar_label(bars_t, fmt="%.4f", padding=3, fontsize=7)
    ax_t.set_xlabel("TF-IDF 평균")
    ax_t.set_title(f"전체 TF-IDF 상위 {len(t30)}개 단어", fontsize=13, fontweight="bold", pad=10)
    ax_t.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig_t)
    plt.close(fig_t)

    # 데이터 테이블
    st.dataframe(
        tfidf_top,
        column_config={
            "순위":       st.column_config.NumberColumn(width="small"),
            "TF-IDF 평균": st.column_config.NumberColumn(format="%.6f"),
        },
        use_container_width=True,
        hide_index=True,
        height=380,
    )

    # 전체 TF-IDF 워드 클라우드
    st.markdown("##### ☁️ 전체 TF-IDF 워드 클라우드")
    try:
        wc_t_all = WordCloud(
            font_path=FONT_PATH if FONT_PATH else None,
            background_color="white",
            width=1100, height=550,
            max_words=100,
            colormap=wc_cmap,
            prefer_horizontal=0.9,
        ).generate_from_frequencies(
            dict(zip(tfidf_all_df["단어"].head(100), tfidf_all_df["TF-IDF 평균"].head(100)))
        )
        fig_twc, ax_twc = plt.subplots(figsize=(13, 6))
        ax_twc.imshow(wc_t_all, interpolation="bilinear")
        ax_twc.axis("off")
        plt.tight_layout(pad=0)
        st.pyplot(fig_twc)
        plt.close(fig_twc)

        buf_tall = io.BytesIO()
        wc_t_all.to_image().save(buf_tall, format="PNG")
        st.download_button(
            label="🖼 전체 TF-IDF 워드 클라우드 PNG",
            data=buf_tall.getvalue(),
            file_name="tfidf_워드클라우드_전체.png",
            mime="image/png",
        )
    except Exception as e:
        st.error(f"전체 TF-IDF 워드 클라우드 오류: {e}")

    st.divider()

    # ══════════════════════════════════════════════════
    # 섹션 B: 연도별 TF-IDF
    # ══════════════════════════════════════════════════
    st.markdown("#### 📅 연도별 TF-IDF 분석")

    tfidf_yearly_targets = sorted(
        df_work[year_col].dropna().astype(str).str.strip().unique().tolist()
    )

    with st.spinner(f"연도별 TF-IDF 계산 중… ({len(tfidf_yearly_targets)}개 연도)"):
        yearly_tfidf   = {}   # {year: [(word, score), ...]}
        tfidf_yr_rows  = []   # 연도별 상위30 CSV용
        tfidf_yr_pivot = []   # 피벗 CSV용

        for yr in tfidf_yearly_targets:
            yr_mask  = df_work[year_col].astype(str).str.strip() == yr
            yr_texts = df_work[yr_mask][text_col].dropna().astype(str).tolist()
            yr_docs  = texts_to_noun_docs(kiwi, yr_texts, stopwords_set, min_len)
            yr_valid = [d for d in yr_docs if d.strip()]

            if len(yr_valid) < 2:
                continue

            vec_yr  = TfidfVectorizer(max_features=100)
            mat_yr  = vec_yr.fit_transform(yr_valid)
            feat_yr = vec_yr.get_feature_names_out()
            mean_yr = mat_yr.mean(axis=0).A1

            yr_sorted = sorted(zip(feat_yr, mean_yr), key=lambda x: x[1], reverse=True)
            yearly_tfidf[yr] = yr_sorted

            for rank, (word, score) in enumerate(yr_sorted[:30], 1):
                tfidf_yr_rows.append({"연도": yr, "순위": rank, "단어": word, "TF-IDF": round(score, 6)})
            for word, score in yr_sorted:
                tfidf_yr_pivot.append({"연도": yr, "단어": word, "TF-IDF": round(score, 6)})

    # 다운로드 버튼
    if tfidf_yr_rows:
        tfidf_yr_df = pd.DataFrame(tfidf_yr_rows)
        tfidf_pv_df = (
            pd.DataFrame(tfidf_yr_pivot)
            .pivot_table(index="단어", columns="연도", values="TF-IDF", fill_value=0)
            .assign(평균=lambda d: d.mean(axis=1))
            .sort_values("평균", ascending=False)
            .round(6)
        )

        dl_yr1, dl_yr2 = st.columns(2)
        with dl_yr1:
            st.download_button(
                label="⬇ 연도별 TF-IDF 상위 30 CSV",
                data=tfidf_yr_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
                file_name="tfidf_연도별_상위30.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with dl_yr2:
            st.download_button(
                label="⬇ 연도별 TF-IDF 피벗 CSV",
                data=tfidf_pv_df.to_csv(encoding="utf-8-sig").encode("utf-8-sig"),
                file_name="tfidf_연도별_피벗.csv",
                mime="text/csv",
                use_container_width=True,
            )

        st.divider()

        # 연도별 TF-IDF 막대 그래프
        st.markdown("##### 📊 연도별 TF-IDF 상위 15개 단어")
        n_cols = 2
        yr_chunks = [tfidf_yearly_targets[i:i+n_cols] for i in range(0, len(tfidf_yearly_targets), n_cols)]

        for chunk in yr_chunks:
            cols = st.columns(n_cols)
            for ci, yr in enumerate(chunk):
                if yr not in yearly_tfidf:
                    with cols[ci]:
                        st.caption(f"{yr}년: 데이터 부족")
                    continue
                top15 = yearly_tfidf[yr][:15]
                words = [w for w, _ in top15][::-1]
                scores = [s for _, s in top15][::-1]

                fig_ty, ax_ty = plt.subplots(figsize=(5, max(3, len(words) * 0.35)))
                bars_ty = ax_ty.barh(words, scores, color="#8b5cf6", alpha=0.82)
                ax_ty.bar_label(bars_ty, fmt="%.4f", padding=3, fontsize=7)
                ax_ty.set_title(f"{yr}년 TF-IDF 상위 {len(words)}개", fontsize=11, fontweight="bold")
                ax_ty.set_xlim(0, max(scores) * 1.25)
                ax_ty.spines[["top", "right"]].set_visible(False)
                ax_ty.tick_params(labelsize=8)
                plt.tight_layout()
                with cols[ci]:
                    st.pyplot(fig_ty)
                plt.close(fig_ty)

        st.divider()

        # 연도별 TF-IDF 워드 클라우드 그리드
        st.markdown("##### ☁️ 연도별 TF-IDF 워드 클라우드")
        wc_cols = st.columns(n_cols)

        for ci, yr in enumerate(tfidf_yearly_targets):
            if yr not in yearly_tfidf:
                continue
            yr_freq_dict = {w: s for w, s in yearly_tfidf[yr][:wc_max]}
            if not yr_freq_dict:
                continue
            try:
                wc_ty = WordCloud(
                    font_path=FONT_PATH if FONT_PATH else None,
                    background_color="white",
                    width=700, height=420,
                    max_words=wc_max,
                    colormap=wc_cmap,
                    prefer_horizontal=0.9,
                ).generate_from_frequencies(yr_freq_dict)

                fig_wty, ax_wty = plt.subplots(figsize=(6, 3.5))
                ax_wty.imshow(wc_ty, interpolation="bilinear")
                ax_wty.axis("off")
                ax_wty.set_title(f"{yr}년 TF-IDF 키워드", fontsize=11, fontweight="bold", pad=6)
                plt.tight_layout(pad=0.5)

                with wc_cols[ci % n_cols]:
                    st.pyplot(fig_wty)
                    buf_ty = io.BytesIO()
                    wc_ty.to_image().save(buf_ty, format="PNG")
                    st.download_button(
                        label=f"🖼 {yr}년 TF-IDF PNG",
                        data=buf_ty.getvalue(),
                        file_name=f"tfidf_워드클라우드_{yr}.png",
                        mime="image/png",
                        use_container_width=True,
                    )
                plt.close(fig_wty)
            except Exception as e:
                with wc_cols[ci % n_cols]:
                    st.error(f"{yr}년 오류: {e}")

        st.divider()

        # 연도별 TF-IDF 통합 워드 클라우드
        st.markdown("##### ☁️ 전체 기간 TF-IDF 통합 워드 클라우드")
        combined_tfidf: dict[str, list] = {}
        for yr, pairs in yearly_tfidf.items():
            for word, score in pairs:
                combined_tfidf.setdefault(word, []).append(score)
        combined_mean = {w: sum(v)/len(v) for w, v in combined_tfidf.items()}
        top_combined  = dict(sorted(combined_mean.items(), key=lambda x: x[1], reverse=True)[:100])

        try:
            wc_tcomb = WordCloud(
                font_path=FONT_PATH if FONT_PATH else None,
                background_color="white",
                width=1100, height=600,
                max_words=100,
                colormap=wc_cmap,
                prefer_horizontal=0.9,
            ).generate_from_frequencies(top_combined)

            fig_tcomb, ax_tcomb = plt.subplots(figsize=(13, 6))
            ax_tcomb.imshow(wc_tcomb, interpolation="bilinear")
            ax_tcomb.axis("off")
            yr_range = f"{min(tfidf_yearly_targets)}~{max(tfidf_yearly_targets)}"
            ax_tcomb.set_title(f"{yr_range}년 TF-IDF 전체 키워드", fontsize=14, fontweight="bold", pad=8)
            plt.tight_layout(pad=0.5)
            st.pyplot(fig_tcomb)
            plt.close(fig_tcomb)

            buf_tcomb = io.BytesIO()
            wc_tcomb.to_image().save(buf_tcomb, format="PNG")
            st.download_button(
                label="🖼 TF-IDF 통합 워드 클라우드 PNG",
                data=buf_tcomb.getvalue(),
                file_name="tfidf_워드클라우드_전체.png",
                mime="image/png",
            )
        except Exception as e:
            st.error(f"통합 TF-IDF 워드 클라우드 오류: {e}")
    else:
        st.warning("연도별 TF-IDF 분석을 위한 충분한 데이터가 없습니다.")

# ════════════════════════════════════════════════════════
# STEP 7: 동시출현 네트워크 탭
# ════════════════════════════════════════════════════════
with tab_cooc:

    # ── 사이드바 추가 설정 읽기 ─────────────────────────
    with st.sidebar:
        st.divider()
        st.markdown("**동시출현 네트워크 설정**")
        cooc_top_words = st.slider("분석 대상 상위 단어 수", 30, 200, 100, key="cooc_top")
        cooc_top_edges = st.slider("네트워크 엣지 수 (전체)",  50, 300, 150, key="cooc_edges")
        cooc_core_edges= st.slider("핵심 네트워크 엣지 수",   20, 100,  50, key="cooc_core")

    # ── 문서별 고유 명사 추출 ────────────────────────────
    with st.spinner("🕸️ 동시출현 분석을 위한 명사 추출 중…"):
        doc_noun_sets = []
        for text in texts:
            try:
                result = kiwi.analyze(str(text))
                nouns = {tok for tok, tag, _, _ in result[0][0]
                         if tag in NOUN_TAGS and len(tok) >= min_len and tok not in stopwords_set}
                doc_noun_sets.append(nouns)
            except Exception:
                doc_noun_sets.append(set())

    # 전체 빈도 기준 상위 단어 집합
    all_flat = [w for ns in doc_noun_sets for w in ns]
    word_freq_all = Counter(all_flat)
    top_word_set  = {w for w, _ in word_freq_all.most_common(cooc_top_words)}

    # ── 동시출현 카운팅 ──────────────────────────────────
    with st.spinner("동시출현 쌍 계산 중…"):
        cooc_counter = Counter()
        for nouns in doc_noun_sets:
            filtered = sorted(nouns & top_word_set)
            for pair in combinations(filtered, 2):
                cooc_counter[pair] += 1

    if not cooc_counter:
        st.warning("동시출현 쌍이 없습니다. 분석 대상 단어 수를 늘리거나 불용어를 줄여보세요.")

    # ── 엣지 데이터프레임 & CSV 다운로드 ─────────────────
    edge_rows = [{"단어1": p[0], "단어2": p[1], "동시출현 빈도": c}
                 for p, c in cooc_counter.most_common()]
    edge_df_cooc = pd.DataFrame(edge_rows)

    st.markdown(f"**총 동시출현 쌍: {len(edge_df_cooc):,}개**")

    dl_cooc1, dl_cooc2 = st.columns([1, 3])
    with dl_cooc1:
        st.download_button(
            label="⬇ 엣지 리스트 CSV",
            data=edge_df_cooc.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
            file_name="cooccurrence_edge_list.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # 상위 30 테이블
    st.markdown("##### 상위 30개 동시출현 쌍")
    top30_cooc = edge_df_cooc.head(30).copy()
    top30_cooc.insert(0, "순위", range(1, len(top30_cooc) + 1))
    st.dataframe(top30_cooc, use_container_width=True, hide_index=True, height=320)

    st.divider()

    # ── 그래프 공통 구성 ─────────────────────────────────
    with st.spinner("네트워크 그래프 생성 중…"):

        # 전체 그래프 (spring layout용)
        G = nx.Graph()
        for (w1, w2), wt in cooc_counter.most_common(cooc_top_edges):
            G.add_edge(w1, w2, weight=wt)

        pos        = nx.spring_layout(G, k=2.5, iterations=80, seed=42)
        e_widths   = [G[u][v]["weight"] * 0.03 for u, v in G.edges()]
        node_deg   = [G.degree(n) for n in G.nodes()]

        # 커뮤니티 탐지
        try:
            from networkx.algorithms.community import louvain_communities
            communities = louvain_communities(G, seed=42)
        except Exception:
            from networkx.algorithms.community import greedy_modularity_communities
            communities = list(greedy_modularity_communities(G))

        node_comm = {}
        for ci, comm in enumerate(communities):
            for nd in comm:
                node_comm[nd] = ci
        comm_colors = [node_comm.get(n, 0) for n in G.nodes()]

        degree_cent = nx.degree_centrality(G)

    # ────────────────────────────────────────────────────
    # 네트워크 그래프 선택 탭
    # ────────────────────────────────────────────────────
    net_tab1, net_tab2, net_tab3, net_tab4, net_tab5 = st.tabs([
        "기본 네트워크", "연결 중심성", "커뮤니티", "핵심 네트워크", "원형 레이아웃"
    ])

    # 1. 기본 네트워크
    with net_tab1:
        fig1, ax1 = plt.subplots(figsize=(14, 10))
        nx.draw_networkx_edges(G, pos, ax=ax1, alpha=0.4, width=e_widths, edge_color="gray")
        nx.draw_networkx_nodes(G, pos, ax=ax1,
                               node_size=[d * 150 for d in node_deg],
                               node_color="skyblue", alpha=0.85,
                               edgecolors="darkblue", linewidths=1)
        nx.draw_networkx_labels(G, pos, ax=ax1, font_size=9, font_weight="bold", font_family=FONT_NAME)
        ax1.set_title("동시출현 단어 네트워크", fontsize=16, fontweight="bold")
        ax1.axis("off")
        plt.tight_layout()
        st.pyplot(fig1)
        st.download_button("🖼 기본 네트워크 PNG",
                           save_fig_png(fig1), "network_basic.png", "image/png")
        plt.close(fig1)

    # 2. 연결 중심성
    with net_tab2:
        cent_vals  = [degree_cent[n] for n in G.nodes()]
        node_size2 = [degree_cent[n] * 3000 + 200 for n in G.nodes()]

        fig2, ax2 = plt.subplots(figsize=(14, 10))
        nx.draw_networkx_edges(G, pos, ax=ax2, alpha=0.3, width=e_widths, edge_color="lightgray")
        nodes2 = nx.draw_networkx_nodes(G, pos, ax=ax2,
                                        node_size=node_size2, node_color=cent_vals,
                                        cmap=plt.cm.YlOrRd, alpha=0.9,
                                        edgecolors="black", linewidths=1)
        nx.draw_networkx_labels(G, pos, ax=ax2, font_size=9, font_weight="bold", font_family=FONT_NAME)
        plt.colorbar(nodes2, ax=ax2, label="연결 중심성", shrink=0.8)
        ax2.set_title("동시출현 네트워크 — 연결 중심성", fontsize=16, fontweight="bold")
        ax2.axis("off")
        plt.tight_layout()
        st.pyplot(fig2)
        st.download_button("🖼 연결 중심성 PNG",
                           save_fig_png(fig2), "network_centrality.png", "image/png")
        plt.close(fig2)

        # 중심성 테이블
        st.markdown("##### 연결 중심성 상위 20개 단어")
        cent_df = (
            pd.DataFrame({"단어": list(degree_cent.keys()),
                          "연결 중심성": list(degree_cent.values()),
                          "연결 수": [G.degree(n) for n in degree_cent]})
            .sort_values("연결 중심성", ascending=False)
            .head(20)
            .reset_index(drop=True)
        )
        cent_df.insert(0, "순위", range(1, len(cent_df) + 1))
        cent_df["연결 중심성"] = cent_df["연결 중심성"].round(4)
        st.dataframe(cent_df, use_container_width=True, hide_index=True)

    # 3. 커뮤니티
    with net_tab3:
        fig3, ax3 = plt.subplots(figsize=(14, 10))
        nx.draw_networkx_edges(G, pos, ax=ax3, alpha=0.3, width=e_widths, edge_color="lightgray")
        nx.draw_networkx_nodes(G, pos, ax=ax3,
                               node_size=[G.degree(n) * 150 for n in G.nodes()],
                               node_color=comm_colors, cmap=plt.cm.Set1,
                               alpha=0.9, edgecolors="black", linewidths=1)
        nx.draw_networkx_labels(G, pos, ax=ax3, font_size=9, font_weight="bold", font_family=FONT_NAME)
        ax3.set_title(f"동시출현 네트워크 — 커뮤니티 탐지 ({len(communities)}개 그룹)",
                      fontsize=16, fontweight="bold")
        ax3.axis("off")
        plt.tight_layout()
        st.pyplot(fig3)
        st.download_button("🖼 커뮤니티 네트워크 PNG",
                           save_fig_png(fig3), "network_community.png", "image/png")
        plt.close(fig3)

        # 커뮤니티별 주요 단어
        st.markdown("##### 커뮤니티별 주요 단어 (연결 수 기준 상위 5개)")
        comm_rows = []
        for ci, comm in enumerate(communities):
            top5 = sorted(comm, key=lambda x: G.degree(x), reverse=True)[:5]
            comm_rows.append({"그룹": f"그룹 {ci+1}", "단어 수": len(comm),
                              "주요 단어": ", ".join(top5)})
        st.dataframe(pd.DataFrame(comm_rows), use_container_width=True, hide_index=True)

    # 4. 핵심 네트워크
    with net_tab4:
        G_core = nx.Graph()
        for (w1, w2), wt in cooc_counter.most_common(cooc_core_edges):
            G_core.add_edge(w1, w2, weight=wt)

        pos_core  = nx.spring_layout(G_core, k=3, iterations=100, seed=42)
        ew_core   = [G_core[u][v]["weight"] * 0.05 for u, v in G_core.edges()]
        ns_core   = [G_core.degree(n) * 300 for n in G_core.nodes()]
        el_core   = {(u, v): G_core[u][v]["weight"] for u, v in G_core.edges()}

        fig4, ax4 = plt.subplots(figsize=(13, 9))
        nx.draw_networkx_edges(G_core, pos_core, ax=ax4, alpha=0.55,
                               width=ew_core, edge_color="steelblue")
        nx.draw_networkx_nodes(G_core, pos_core, ax=ax4,
                               node_size=ns_core, node_color="lightcoral",
                               alpha=0.9, edgecolors="darkred", linewidths=2)
        nx.draw_networkx_labels(G_core, pos_core, ax=ax4, font_size=10, font_weight="bold", font_family=FONT_NAME)
        nx.draw_networkx_edge_labels(G_core, pos_core, edge_labels=el_core,
                                     ax=ax4, font_size=7)
        ax4.set_title(f"핵심 동시출현 네트워크 (상위 {cooc_core_edges}개 연결)",
                      fontsize=16, fontweight="bold")
        ax4.axis("off")
        plt.tight_layout()
        st.pyplot(fig4)
        st.download_button("🖼 핵심 네트워크 PNG",
                           save_fig_png(fig4), "network_core.png", "image/png")
        plt.close(fig4)

    # 5. 원형 레이아웃
    with net_tab5:
        pos_circ = nx.circular_layout(G)

        fig5, ax5 = plt.subplots(figsize=(14, 14))
        nx.draw_networkx_edges(G, pos_circ, ax=ax5, alpha=0.2, width=0.5, edge_color="gray")
        nx.draw_networkx_nodes(G, pos_circ, ax=ax5,
                               node_size=[G.degree(n) * 100 for n in G.nodes()],
                               node_color=comm_colors, cmap=plt.cm.Set2,
                               alpha=0.9, edgecolors="black", linewidths=1)
        nx.draw_networkx_labels(G, pos_circ, ax=ax5, font_size=8, font_family=FONT_NAME)
        ax5.set_title("동시출현 네트워크 — 원형 레이아웃", fontsize=16, fontweight="bold")
        ax5.axis("off")
        plt.tight_layout()
        st.pyplot(fig5)
        st.download_button("🖼 원형 레이아웃 PNG",
                           save_fig_png(fig5), "network_circular.png", "image/png")
        plt.close(fig5)

    st.divider()

    # ════════════════════════════════════════════════════
    # 연도별 동시출현 분석 (5_cooccurrence_yearly)
    # ════════════════════════════════════════════════════
    st.markdown("#### 📅 연도별 동시출현 분석")

    cooc_yr_targets = sorted(
        df_work[year_col].dropna().astype(str).str.strip().unique().tolist()
    )

    with st.spinner(f"연도별 동시출현 계산 중… ({len(cooc_yr_targets)}개 연도)"):
        yearly_cooc   = {}        # {year: Counter}
        yr_edge_rows  = []        # 연도별 엣지 CSV용
        yr_top10_rows = []        # 연도별 상위10 CSV용

        for yr in cooc_yr_targets:
            yr_mask  = df_work[year_col].astype(str).str.strip() == yr
            yr_texts = df_work[yr_mask][text_col].dropna().astype(str).tolist()

            yr_noun_sets = []
            for txt in yr_texts:
                try:
                    res = kiwi.analyze(str(txt))
                    ns  = {tok for tok, tag, _, _ in res[0][0]
                           if tag in NOUN_TAGS and len(tok) >= min_len and tok not in stopwords_set}
                    yr_noun_sets.append(ns)
                except Exception:
                    yr_noun_sets.append(set())

            yr_cooc = Counter()
            for ns in yr_noun_sets:
                filtered = sorted(ns & top_word_set)
                for pair in combinations(filtered, 2):
                    yr_cooc[pair] += 1

            yearly_cooc[yr] = yr_cooc

            for (w1, w2), cnt in yr_cooc.most_common(100):
                yr_edge_rows.append({"연도": yr, "source": w1, "target": w2, "weight": cnt})
            for rank, ((w1, w2), cnt) in enumerate(yr_cooc.most_common(10), 1):
                yr_top10_rows.append({"연도": yr, "순위": rank,
                                      "단어1": w1, "단어2": w2, "빈도": cnt})

    # ── 다운로드 버튼 ────────────────────────────────────
    yr_edge_df  = pd.DataFrame(yr_edge_rows)
    yr_top10_df = pd.DataFrame(yr_top10_rows)

    dl_ye1, dl_ye2 = st.columns(2)
    with dl_ye1:
        st.download_button(
            "⬇ 연도별 엣지 리스트 CSV",
            yr_edge_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
            "cooccurrence_yearly_edges.csv", "text/csv", use_container_width=True,
        )
    with dl_ye2:
        st.download_button(
            "⬇ 연도별 상위 10 쌍 CSV",
            yr_top10_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
            "cooccurrence_yearly_top10.csv", "text/csv", use_container_width=True,
        )

    st.divider()

    # ── 연도별 개별 네트워크 (2열 그리드) ────────────────
    st.markdown("##### 연도별 네트워크 그래프")
    n_cols2 = 2
    yr_chunks2 = [cooc_yr_targets[i:i+n_cols2] for i in range(0, len(cooc_yr_targets), n_cols2)]

    for chunk in yr_chunks2:
        cols2 = st.columns(n_cols2)
        for ci, yr in enumerate(chunk):
            yr_cooc = yearly_cooc.get(yr, Counter())
            if len(yr_cooc) < 5:
                with cols2[ci]:
                    st.caption(f"{yr}년: 데이터 부족")
                continue

            Gy = nx.Graph()
            for (w1, w2), wt in yr_cooc.most_common(50):
                Gy.add_edge(w1, w2, weight=wt)
            if len(Gy.nodes()) < 3:
                continue

            pos_y    = nx.spring_layout(Gy, k=2, iterations=50, seed=42)
            ns_y     = [Gy.degree(n) * 150 for n in Gy.nodes()]
            ew_y     = [Gy[u][v]["weight"] * 0.1 for u, v in Gy.edges()]
            dc_y     = nx.degree_centrality(Gy)
            nc_y     = [dc_y[n] for n in Gy.nodes()]

            fig_y, ax_y = plt.subplots(figsize=(6, 5))
            nx.draw_networkx_edges(Gy, pos_y, ax=ax_y, alpha=0.4, width=ew_y, edge_color="gray")
            nd_y = nx.draw_networkx_nodes(Gy, pos_y, ax=ax_y, node_size=ns_y,
                                          node_color=nc_y, cmap=plt.cm.YlOrRd,
                                          alpha=0.9, edgecolors="black", linewidths=1)
            nx.draw_networkx_labels(Gy, pos_y, ax=ax_y, font_size=8, font_weight="bold", font_family=FONT_NAME)
            plt.colorbar(nd_y, ax=ax_y, label="연결 중심성", shrink=0.7)
            ax_y.set_title(f"{yr}년 동시출현 네트워크", fontsize=11, fontweight="bold")
            ax_y.axis("off")
            plt.tight_layout()
            with cols2[ci]:
                st.pyplot(fig_y)
                st.download_button(f"🖼 {yr}년 PNG", save_fig_png(fig_y),
                                   f"network_{yr}.png", "image/png",
                                   use_container_width=True)
            plt.close(fig_y)

    st.divider()

    # ── 연도별 비교 그리드 ────────────────────────────────
    st.markdown("##### 연도별 동시출현 네트워크 비교")
    n_grid_cols = min(4, len(cooc_yr_targets))
    n_grid_rows = -(-len(cooc_yr_targets) // n_grid_cols)   # ceiling division

    fig_cmp, axes_cmp = plt.subplots(n_grid_rows, n_grid_cols,
                                     figsize=(n_grid_cols * 6, n_grid_rows * 5))
    axes_flat = axes_cmp.flatten() if hasattr(axes_cmp, "flatten") else [axes_cmp]

    for idx, yr in enumerate(cooc_yr_targets):
        ax_c = axes_flat[idx]
        yr_cooc = yearly_cooc.get(yr, Counter())
        if len(yr_cooc) < 5:
            ax_c.text(0.5, 0.5, f"{yr}년\n데이터 부족",
                      ha="center", va="center", fontsize=12)
            ax_c.axis("off")
            continue

        Gc = nx.Graph()
        for (w1, w2), wt in yr_cooc.most_common(30):
            Gc.add_edge(w1, w2, weight=wt)
        if len(Gc.nodes()) < 3:
            ax_c.axis("off")
            continue

        pos_c = nx.spring_layout(Gc, k=2, iterations=30, seed=42)
        nx.draw_networkx_edges(Gc, pos_c, ax=ax_c, alpha=0.3, width=0.5, edge_color="gray")
        nx.draw_networkx_nodes(Gc, pos_c, ax=ax_c,
                               node_size=[Gc.degree(n) * 80 for n in Gc.nodes()],
                               node_color="coral", alpha=0.85,
                               edgecolors="darkred", linewidths=0.5)
        nx.draw_networkx_labels(Gc, pos_c, ax=ax_c, font_size=7, font_weight="bold", font_family=FONT_NAME)
        ax_c.set_title(f"{yr}년", fontsize=13, fontweight="bold")
        ax_c.axis("off")

    for idx in range(len(cooc_yr_targets), len(axes_flat)):
        axes_flat[idx].axis("off")

    yr_range2 = f"{min(cooc_yr_targets)}~{max(cooc_yr_targets)}"
    fig_cmp.suptitle(f"연도별 동시출현 네트워크 비교 ({yr_range2})",
                     fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    st.pyplot(fig_cmp)
    st.download_button("🖼 연도별 비교 그리드 PNG", save_fig_png(fig_cmp),
                       "network_yearly_comparison.png", "image/png")
    plt.close(fig_cmp)

    st.divider()

    # ── 주요 단어 쌍 연도별 트렌드 ───────────────────────
    st.markdown("##### 주요 단어 쌍 연도별 트렌드")

    total_cooc_all = Counter()
    for cnt in yearly_cooc.values():
        total_cooc_all.update(cnt)
    top_pairs_trend = [pair for pair, _ in total_cooc_all.most_common(10)]

    trend_rows = []
    for pair in top_pairs_trend:
        for yr in cooc_yr_targets:
            trend_rows.append({
                "단어쌍": f"{pair[0]}-{pair[1]}",
                "연도": yr,
                "빈도": yearly_cooc[yr].get(pair, 0),
            })
    trend_df = pd.DataFrame(trend_rows)

    fig_tr, ax_tr = plt.subplots(figsize=(12, 6))
    for pair in top_pairs_trend[:6]:
        pname = f"{pair[0]}-{pair[1]}"
        d = trend_df[trend_df["단어쌍"] == pname]
        ax_tr.plot(d["연도"], d["빈도"], marker="o", linewidth=2, markersize=6, label=pname)

    ax_tr.set_xlabel("연도", fontsize=11)
    ax_tr.set_ylabel("동시출현 빈도", fontsize=11)
    ax_tr.set_title("주요 단어 쌍 연도별 동시출현 트렌드", fontsize=14, fontweight="bold")
    ax_tr.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    ax_tr.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig_tr)

    # 트렌드 CSV 다운로드
    trend_pivot = trend_df.pivot(index="연도", columns="단어쌍", values="빈도").fillna(0).astype(int)
    dl_trend1, dl_trend2 = st.columns([1, 3])
    with dl_trend1:
        st.download_button(
            "⬇ 트렌드 CSV",
            trend_pivot.to_csv(encoding="utf-8-sig").encode("utf-8-sig"),
            "cooccurrence_trend.csv", "text/csv", use_container_width=True,
        )
    plt.close(fig_tr)

