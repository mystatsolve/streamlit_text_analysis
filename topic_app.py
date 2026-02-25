# -*- coding: utf-8 -*-
"""
LDA & BERTopic 토픽 분석 전용 Streamlit 앱
"""

import os
import io
import re
import warnings

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import plotly.express as px
import plotly.graph_objects as go

# ── 페이지 설정 ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="토픽 분석 도구",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 한국어 폰트 설정 ─────────────────────────────────────────────────────────
def get_korean_font():
    candidates = [
        r"C:\Windows\Fonts\malgun.ttf",
        r"C:\Windows\Fonts\malgunbd.ttf",
        r"C:\Windows\Fonts\gulim.ttc",
        r"C:\Windows\Fonts\batang.ttc",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    found = fm.findSystemFonts()
    for f in found:
        fn = os.path.basename(f).lower()
        if any(k in fn for k in ["malgun", "nanum", "gulim", "batang", "dotum"]):
            return f
    return None

_font_path = get_korean_font()
FONT_NAME = "sans-serif"
if _font_path:
    fm.fontManager.addfont(_font_path)
    _prop = fm.FontProperties(fname=_font_path)
    FONT_NAME = _prop.get_name()
    plt.rcParams["font.family"] = FONT_NAME
    plt.rcParams["font.sans-serif"] = [FONT_NAME] + plt.rcParams.get("font.sans-serif", [])
plt.rcParams["axes.unicode_minus"] = False

# ── LDA 임포트 ───────────────────────────────────────────────────────────────
try:
    from gensim import corpora
    from gensim.models import LdaModel
    import pyLDAvis
    import pyLDAvis.gensim_models as gensimvis
    HAS_LDA = True
except ImportError:
    HAS_LDA = False

# ── Kiwi 캐시 ────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="형태소 분석기 로드 중...")
def load_kiwi():
    from kiwipiepy import Kiwi
    return Kiwi()

# ── 유틸 함수 ────────────────────────────────────────────────────────────────
DEFAULT_STOPWORDS = [
    "있다", "하다", "되다", "이다", "없다", "않다", "같다", "보다", "나다",
    "것", "들", "그", "수", "이", "등", "때", "년", "가", "를", "로",
    "에", "의", "은", "는", "과", "와", "다", "을", "한", "대", "으로",
    "하는", "위", "및", "또", "또는", "더", "여", "중", "통해", "위해",
    "따라", "따른", "대한", "관한", "통한", "기반", "본", "이러한", "매우",
    "데", "그리고", "그러나", "따라서", "하지만", "그래서", "또한", "즉",
    "뿐", "각", "후", "바", "때문", "않", "못", "이를", "그것", "여기",
    "연구", "논문", "분석", "결과", "방법", "목적", "대상", "사용", "활용",
    "제시", "확인", "진행", "수행", "내용", "필요", "가능", "제공", "개발",
    "구축", "설계", "적용", "도입", "운영", "실시", "조사", "검토", "파악",
    "살펴", "본고", "본문", "논의", "고찰", "탐색", "탐구", "검증", "평가",
    "도출", "시사점", "제언", "주요", "측면", "과정", "영역", "부분", "정도",
    "경우", "이상", "이하", "사이", "정보", "자료", "기술", "시스템",
    "study", "research", "paper", "results", "analysis", "method",
    "data", "based", "using", "used", "this", "that", "the", "and", "for",
]

def fig_to_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="PNG", dpi=150, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    return buf.getvalue()

def df_to_csv_bytes(df):
    return df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")

def detect_columns(df):
    """abstract/year 컬럼 자동 감지"""
    text_col, year_col = None, None
    for c in df.columns:
        cl = c.lower()
        if text_col is None and any(k in cl for k in ["abstract", "text", "sentence", "내용", "본문", "초록"]):
            text_col = c
        if year_col is None and any(k in cl for k in ["year", "연도", "년도", "년"]):
            year_col = c
    if text_col is None and len(df.columns) > 0:
        text_col = df.columns[0]
    if year_col is None and len(df.columns) > 1:
        year_col = df.columns[1]
    return text_col, year_col

def extract_nouns(text, kiwi, stopwords_set, min_len=2):
    """Kiwi로 명사 추출 (LDA용 list 반환)"""
    if pd.isna(text) or not str(text).strip():
        return []
    text = re.sub(r"[a-zA-Z]+", " ", str(text))
    text = re.sub(r"[^가-힣\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    try:
        result = kiwi.analyze(text)
        nouns = []
        for token, pos, _, _ in result[0][0]:
            if pos in ("NNG", "NNP") and len(token) >= min_len and token not in stopwords_set:
                nouns.append(token)
        return nouns
    except Exception:
        return []

def extract_nouns_str(text, kiwi, stopwords_set, min_len=2):
    """Kiwi로 명사 추출 (BERTopic용 공백 구분 문자열 반환)"""
    tokens = extract_nouns(text, kiwi, stopwords_set, min_len)
    return " ".join(tokens)

# ── 사이드바 ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔬 토픽 분석 도구")
    st.markdown("---")

    st.subheader("📂 데이터 파일")
    uploaded = st.file_uploader("CSV 파일 업로드", type=["csv"])

    # 기본 경로 후보
    DEFAULT_DATA_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "7_topic", "abstract_year_data.csv"
    )
    use_default = st.checkbox(
        "기본 데이터 사용 (abstract_year_data.csv)",
        value=(uploaded is None),
        disabled=(uploaded is not None),
    )

    st.markdown("---")
    st.subheader("⚙️ 공통 설정")
    min_word_len = st.slider("최소 단어 길이", 1, 4, 2)

    with st.expander("불용어 편집"):
        sw_text = st.text_area(
            "불용어 목록 (쉼표 또는 줄바꿈으로 구분)",
            value=", ".join(DEFAULT_STOPWORDS),
            height=200,
        )
    stopwords_set = set(
        w.strip() for w in re.split(r"[,\n]", sw_text) if w.strip()
    )

    st.markdown("---")
    st.caption("LDA: gensim + pyLDAvis\nBERTopic: sentence-transformers + HDBSCAN")

# ── 데이터 로드 ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_csv(content_bytes: bytes, filename: str):
    for enc in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            return pd.read_csv(io.BytesIO(content_bytes), encoding=enc)
        except Exception:
            continue
    return None

df_raw = None
load_error = None

if uploaded is not None:
    df_raw = load_csv(uploaded.read(), uploaded.name)
    if df_raw is None:
        load_error = "파일을 읽을 수 없습니다. 인코딩을 확인하세요."
elif use_default:
    if os.path.exists(DEFAULT_DATA_PATH):
        try:
            df_raw = pd.read_csv(DEFAULT_DATA_PATH, encoding="utf-8-sig")
        except Exception:
            try:
                df_raw = pd.read_csv(DEFAULT_DATA_PATH, encoding="cp949")
            except Exception as e:
                load_error = f"기본 데이터 로드 실패: {e}"
    else:
        load_error = f"기본 데이터 파일 없음:\n`{DEFAULT_DATA_PATH}`\n\nCSV 파일을 직접 업로드해 주세요."

# ── 컬럼 선택 ────────────────────────────────────────────────────────────────
text_col = year_col = None
if df_raw is not None:
    detected_text, detected_year = detect_columns(df_raw)
    col1, col2 = st.columns(2)
    with col1:
        text_col = st.selectbox("텍스트 컬럼", df_raw.columns.tolist(),
                                index=df_raw.columns.tolist().index(detected_text) if detected_text else 0)
    with col2:
        year_col = st.selectbox("연도 컬럼", df_raw.columns.tolist(),
                                index=df_raw.columns.tolist().index(detected_year) if detected_year else min(1, len(df_raw.columns)-1))

    df_raw[year_col] = pd.to_numeric(df_raw[year_col], errors="coerce")
    df_work = df_raw[[text_col, year_col]].dropna().copy()
    df_work.columns = ["abstract", "year"]
    df_work["year"] = df_work["year"].astype(int)

    st.info(f"📊 총 **{len(df_work):,}**개 문서 · 연도 범위: **{df_work['year'].min()}~{df_work['year'].max()}**")
elif load_error:
    st.error(load_error)
    st.stop()
else:
    st.warning("왼쪽 사이드바에서 CSV 파일을 업로드하거나 기본 데이터를 선택하세요.")
    st.stop()

# ── 탭 ───────────────────────────────────────────────────────────────────────
tab_lda, tab_bert = st.tabs(["🔍 LDA 토픽 분석", "🤖 BERTopic 분석"])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 : LDA
# ════════════════════════════════════════════════════════════════════════════
with tab_lda:
    if not HAS_LDA:
        st.error("gensim 또는 pyLDAvis가 설치되지 않았습니다.\n```\npip install gensim pyLDAvis\n```")
    else:
        st.subheader("🔍 LDA 토픽 분석 설정")
        c1, c2, c3, c4 = st.columns(4)
        lda_n_topics   = c1.slider("토픽 수",        2, 20, 5)
        lda_passes     = c2.slider("Passes",          5, 30, 15)
        lda_no_below   = c3.slider("최소 문서 빈도", 1, 20, 3)
        lda_no_above   = c4.slider("최대 문서 비율", 0.3, 0.99, 0.70, step=0.05, format="%.2f")

        run_lda = st.button("▶ LDA 분석 실행", type="primary", use_container_width=True)

        if run_lda:
            st.session_state.pop("lda_result", None)

            try:
                kiwi = load_kiwi()
            except Exception as e:
                st.error(f"형태소 분석기 로드 실패: {e}")
                st.stop()

            # 명사 추출
            prog = st.progress(0, "명사 추출 중...")
            tokens_list = []
            n = len(df_work)
            for i, row in df_work.iterrows():
                tokens_list.append(extract_nouns(row["abstract"], kiwi, stopwords_set, min_word_len))
                if i % max(1, n // 50) == 0:
                    prog.progress(min(int((df_work.index.get_loc(i) + 1) / n * 40), 40),
                                  f"명사 추출 중... ({df_work.index.get_loc(i)+1}/{n})")

            df_work2 = df_work.copy()
            df_work2["tokens"] = tokens_list
            df_work2 = df_work2[df_work2["tokens"].apply(len) > 0].reset_index(drop=True)

            prog.progress(40, "사전 및 코퍼스 생성 중...")
            dictionary = corpora.Dictionary(df_work2["tokens"].tolist())
            dictionary.filter_extremes(no_below=lda_no_below, no_above=lda_no_above)

            if len(dictionary) < 5:
                st.error("사전 단어가 너무 적습니다. 파라미터(최소 문서 빈도, 최대 문서 비율)를 조정해 보세요.")
                st.stop()

            corpus = [dictionary.doc2bow(t) for t in df_work2["tokens"].tolist()]

            prog.progress(45, f"LDA 모델 학습 중 (토픽 {lda_n_topics}개, passes {lda_passes})...")
            lda_model = LdaModel(
                corpus=corpus,
                id2word=dictionary,
                num_topics=lda_n_topics,
                random_state=42,
                passes=lda_passes,
                alpha="auto",
                eta="auto",
                chunksize=100,
            )

            prog.progress(80, "토픽 할당 및 시각화 준비 중...")

            # 토픽별 키워드
            topics_rows = []
            for idx in range(lda_n_topics):
                top10 = lda_model.show_topic(idx, topn=10)
                keywords = [w for w, _ in top10]
                weights  = [round(s, 4) for _, s in top10]
                topics_rows.append({
                    "토픽": f"토픽 {idx+1}",
                    "주요 키워드 (상위 10개)": ", ".join(keywords),
                    "가중치": ", ".join(str(w) for w in weights),
                })
            topics_df = pd.DataFrame(topics_rows)

            # 문서별 토픽 할당
            doc_topics_rows = []
            for i, bow in enumerate(corpus):
                dist = lda_model.get_document_topics(bow, minimum_probability=0.0)
                dom  = max(dist, key=lambda x: x[1])
                doc_topics_rows.append({
                    "doc_id": i,
                    "year": df_work2.iloc[i]["year"],
                    "dominant_topic": dom[0] + 1,
                    "topic_prob": round(dom[1], 4),
                })
            doc_topics_df = pd.DataFrame(doc_topics_rows)

            # 연도별 토픽 비율
            yearly = doc_topics_df.groupby(["year", "dominant_topic"]).size().unstack(fill_value=0)
            yearly_pct = yearly.div(yearly.sum(axis=1), axis=0) * 100
            yearly_pct.columns = [f"토픽{c}" for c in yearly_pct.columns]
            yearly_pct = yearly_pct.reset_index()

            # pyLDAvis HTML
            prog.progress(90, "pyLDAvis 시각화 생성 중...")
            try:
                vis_data = gensimvis.prepare(lda_model, corpus, dictionary, sort_topics=False)
                lda_html = pyLDAvis.prepared_data_to_html(vis_data)
            except Exception as e:
                lda_html = f"<p style='color:red'>pyLDAvis 생성 실패: {e}</p>"

            prog.progress(100, "완료!")
            prog.empty()

            st.session_state["lda_result"] = {
                "topics_df": topics_df,
                "doc_topics_df": doc_topics_df,
                "yearly_pct": yearly_pct,
                "lda_html": lda_html,
                "n_topics": lda_n_topics,
                "n_docs": len(df_work2),
                "dict_size": len(dictionary),
            }

        # ── 결과 표시 ────────────────────────────────────────────────────
        if "lda_result" in st.session_state:
            res = st.session_state["lda_result"]
            topics_df     = res["topics_df"]
            doc_topics_df = res["doc_topics_df"]
            yearly_pct    = res["yearly_pct"]
            lda_html      = res["lda_html"]

            st.markdown("---")

            # 요약 지표
            m1, m2, m3 = st.columns(3)
            m1.metric("토픽 수",        res["n_topics"])
            m2.metric("분석 문서 수",   f"{res['n_docs']:,}")
            m3.metric("사전 단어 수",   f"{res['dict_size']:,}")

            # 토픽별 키워드 테이블
            st.subheader("📋 토픽별 주요 키워드")
            st.dataframe(topics_df, use_container_width=True, hide_index=True)
            st.download_button("⬇ 토픽 키워드 CSV 다운로드", df_to_csv_bytes(topics_df),
                               "lda_topics.csv", "text/csv")

            st.markdown("---")
            col_a, col_b = st.columns(2)

            # 토픽 분포 바 차트
            with col_a:
                st.subheader("📊 토픽별 문서 분포")
                dist_df = (doc_topics_df["dominant_topic"]
                           .value_counts().sort_index().reset_index())
                dist_df.columns = ["토픽", "문서 수"]
                dist_df["토픽"] = dist_df["토픽"].apply(lambda x: f"토픽 {x}")
                fig_dist = px.bar(dist_df, x="문서 수", y="토픽",
                                  orientation="h", color="문서 수",
                                  color_continuous_scale="Blues",
                                  title="토픽별 문서 수")
                fig_dist.update_layout(yaxis={"categoryorder": "total ascending"},
                                       coloraxis_showscale=False, height=350)
                st.plotly_chart(fig_dist, use_container_width=True)

            # 연도별 토픽 트렌드
            with col_b:
                st.subheader("📅 연도별 토픽 비율 트렌드")
                topic_cols = [c for c in yearly_pct.columns if c != "year"]
                fig_trend = go.Figure()
                for tc in topic_cols:
                    fig_trend.add_trace(go.Bar(
                        x=yearly_pct["year"], y=yearly_pct[tc], name=tc
                    ))
                fig_trend.update_layout(
                    barmode="stack",
                    xaxis_title="연도", yaxis_title="비율 (%)",
                    title="연도별 토픽 비율 (누적 막대)",
                    legend=dict(orientation="h", y=-0.3),
                    height=350,
                )
                st.plotly_chart(fig_trend, use_container_width=True)

            st.markdown("---")

            # 연도별 데이터 다운로드
            st.download_button("⬇ 연도별 토픽 트렌드 CSV", df_to_csv_bytes(yearly_pct),
                               "lda_yearly_trend.csv", "text/csv")
            st.download_button("⬇ 문서별 토픽 할당 CSV", df_to_csv_bytes(doc_topics_df),
                               "lda_document_topics.csv", "text/csv")

            # pyLDAvis 인터랙티브 시각화
            st.subheader("🌐 pyLDAvis 인터랙티브 시각화")
            st.caption("버블을 클릭하면 해당 토픽의 키워드 분포를 확인할 수 있습니다.")
            components.html(lda_html, height=800, scrolling=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 : BERTopic
# ════════════════════════════════════════════════════════════════════════════
with tab_bert:
    st.subheader("🤖 BERTopic 분석 설정")

    c1, c2, c3 = st.columns(3)
    bert_min_cluster = c1.slider("최소 클러스터 크기",  10, 300, 100, step=10)
    bert_min_samples = c2.slider("최소 샘플 수",         3,  50,  15)
    bert_min_topic   = c3.slider("최소 토픽 크기",       10, 300, 100, step=10)

    st.caption("⚠️ 임베딩 모델(paraphrase-multilingual-MiniLM-L12-v2)이 최초 실행 시 자동 다운로드됩니다. 약 5~10분 소요될 수 있습니다.")
    run_bert = st.button("▶ BERTopic 분석 실행", type="primary", use_container_width=True)

    if run_bert:
        st.session_state.pop("bert_result", None)

        with st.spinner("BERTopic 패키지 로드 중..."):
            try:
                from bertopic import BERTopic
                from sentence_transformers import SentenceTransformer
                from sklearn.feature_extraction.text import CountVectorizer as CV
                from hdbscan import HDBSCAN
            except Exception as e:
                st.error(f"BERTopic 임포트 실패:\n```\n{e}\n```")
                st.stop()

        try:
            kiwi = load_kiwi()
        except Exception as e:
            st.error(f"형태소 분석기 로드 실패: {e}")
            st.stop()

        # 명사 추출
        prog = st.progress(0, "명사 추출 중...")
        noun_texts = []
        n = len(df_work)
        for i, row in df_work.iterrows():
            noun_texts.append(extract_nouns_str(row["abstract"], kiwi, stopwords_set, min_word_len))
            loc = df_work.index.get_loc(i)
            if loc % max(1, n // 50) == 0:
                prog.progress(min(int((loc + 1) / n * 30), 30),
                              f"명사 추출 중... ({loc+1}/{n})")

        df_bert = df_work.copy()
        df_bert["nouns"] = noun_texts
        df_bert = df_bert[df_bert["nouns"].str.strip() != ""].reset_index(drop=True)
        docs  = df_bert["nouns"].tolist()
        years = df_bert["year"].astype(int).tolist()

        prog.progress(30, "임베딩 모델 로드 중...")
        try:
            emb_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        except Exception as e:
            st.error(f"임베딩 모델 로드 실패: {e}")
            st.stop()

        prog.progress(40, "BERTopic 모델 학습 중 (시간이 걸립니다)...")
        try:
            vec_model = CV(ngram_range=(1, 2), min_df=2, max_df=1.0)
            hdbscan_model = HDBSCAN(
                min_cluster_size=bert_min_cluster,
                min_samples=bert_min_samples,
                metric="euclidean",
                cluster_selection_method="eom",
                prediction_data=True,
            )
            topic_model = BERTopic(
                embedding_model=emb_model,
                vectorizer_model=vec_model,
                hdbscan_model=hdbscan_model,
                language="multilingual",
                calculate_probabilities=True,
                verbose=False,
                min_topic_size=bert_min_topic,
            )
            topics, probs = topic_model.fit_transform(docs)
        except Exception as e:
            st.error(f"BERTopic 학습 실패:\n```\n{e}\n```")
            st.stop()

        prog.progress(75, "결과 정리 중...")

        topic_info = topic_model.get_topic_info()
        valid_topics = topic_info[topic_info["Topic"] != -1]

        # 토픽별 키워드 DF
        kw_rows = []
        for _, row in valid_topics.iterrows():
            words = topic_model.get_topic(row["Topic"])
            if words:
                kw_rows.append({
                    "토픽 ID": row["Topic"],
                    "문서 수": row["Count"],
                    "상위 키워드 (10개)": ", ".join(w for w, _ in words[:10]),
                })
        kw_df = pd.DataFrame(kw_rows)

        # 문서별 토픽
        df_bert["topic"] = topics
        doc_prob = []
        for p in probs:
            arr = np.asarray(p)
            doc_prob.append(float(arr.max()) if arr.size > 0 else 0.0)
        df_bert["topic_prob"] = doc_prob
        doc_topics_out = df_bert[["abstract", "year", "topic", "topic_prob"]].copy()

        # 연도별 토픽 트렌드 (topics_over_time)
        prog.progress(85, "연도별 트렌드 분석 중...")
        try:
            tot_df = topic_model.topics_over_time(docs, years, nr_bins=min(8, len(set(years))))
        except Exception:
            tot_df = None

        # Plotly 시각화 HTML
        prog.progress(92, "시각화 생성 중...")
        barchart_html = heatmap_html = hierarchy_html = None
        top_n = min(10, len(valid_topics))
        try:
            fig_bc = topic_model.visualize_barchart(top_n_topics=top_n, n_words=8)
            barchart_html = fig_bc.to_html(full_html=False)
        except Exception:
            pass
        try:
            fig_hm = topic_model.visualize_heatmap(top_n_topics=min(15, len(valid_topics)))
            heatmap_html = fig_hm.to_html(full_html=False)
        except Exception:
            pass
        try:
            fig_hi = topic_model.visualize_hierarchy(top_n_topics=min(20, len(valid_topics)))
            hierarchy_html = fig_hi.to_html(full_html=False)
        except Exception:
            pass

        prog.progress(100, "완료!")
        prog.empty()

        st.session_state["bert_result"] = {
            "kw_df": kw_df,
            "topic_info": topic_info,
            "doc_topics_out": doc_topics_out,
            "tot_df": tot_df,
            "barchart_html": barchart_html,
            "heatmap_html": heatmap_html,
            "hierarchy_html": hierarchy_html,
            "n_topics": len(valid_topics),
            "n_noise": int(topic_info[topic_info["Topic"] == -1]["Count"].sum()),
            "n_docs": len(df_bert),
        }

    # ── 결과 표시 ────────────────────────────────────────────────────────
    if "bert_result" in st.session_state:
        res = st.session_state["bert_result"]
        kw_df         = res["kw_df"]
        topic_info    = res["topic_info"]
        doc_topics_out = res["doc_topics_out"]
        tot_df        = res["tot_df"]

        st.markdown("---")

        # 요약 지표
        m1, m2, m3 = st.columns(3)
        m1.metric("발견된 토픽 수",   res["n_topics"])
        m2.metric("분석 문서 수",     f"{res['n_docs']:,}")
        m3.metric("노이즈 문서 수",   f"{res['n_noise']:,}")

        # 토픽별 키워드 테이블
        st.subheader("📋 토픽별 주요 키워드")
        st.dataframe(kw_df, use_container_width=True, hide_index=True)
        st.download_button("⬇ 토픽 키워드 CSV 다운로드", df_to_csv_bytes(kw_df),
                           "bertopic_keywords.csv", "text/csv")
        st.download_button("⬇ 문서별 토픽 할당 CSV", df_to_csv_bytes(doc_topics_out),
                           "bertopic_document_topics.csv", "text/csv")

        st.markdown("---")
        col_a, col_b = st.columns(2)

        # 토픽 분포 바차트
        with col_a:
            st.subheader("📊 토픽별 문서 분포")
            dist_df = (topic_info[topic_info["Topic"] != -1]
                       .sort_values("Count", ascending=False).head(20))
            fig_dist = px.bar(
                dist_df, x="Count", y=dist_df["Topic"].astype(str),
                orientation="h", color="Count",
                color_continuous_scale="Teal",
                labels={"Count": "문서 수", "y": "토픽 ID"},
                title="토픽별 문서 수 (상위 20개)"
            )
            fig_dist.update_layout(yaxis={"categoryorder": "total ascending"},
                                   coloraxis_showscale=False, height=400)
            st.plotly_chart(fig_dist, use_container_width=True)

        # 연도별 트렌드
        with col_b:
            st.subheader("📅 주요 토픽 연도별 트렌드")
            if tot_df is not None and not tot_df.empty:
                top_ids = (topic_info[topic_info["Topic"] != -1]
                           .sort_values("Count", ascending=False).head(6)["Topic"].tolist())
                fig_line = go.Figure()
                for tid in top_ids:
                    sub = tot_df[tot_df["Topic"] == tid]
                    if sub.empty:
                        continue
                    label = f"토픽{tid}: {sub['Words'].iloc[0].split(',')[0]}"
                    fig_line.add_trace(go.Scatter(
                        x=sub["Timestamp"], y=sub["Frequency"],
                        mode="lines+markers", name=label,
                    ))
                fig_line.update_layout(
                    xaxis_title="연도", yaxis_title="빈도",
                    title="주요 토픽 연도별 트렌드 (상위 6개)",
                    legend=dict(orientation="h", y=-0.35),
                    height=400,
                )
                st.plotly_chart(fig_line, use_container_width=True)
                st.download_button("⬇ 연도별 트렌드 CSV", df_to_csv_bytes(tot_df),
                                   "bertopic_topics_over_time.csv", "text/csv")
            else:
                st.info("연도별 트렌드 데이터를 생성할 수 없습니다.")

        st.markdown("---")

        # 토픽 바차트 (키워드)
        if res.get("barchart_html"):
            st.subheader("📊 토픽별 키워드 바차트")
            components.html(
                f"<html><head><meta charset='utf-8'></head><body>{res['barchart_html']}</body></html>",
                height=600, scrolling=True
            )

        # 히트맵
        if res.get("heatmap_html"):
            st.subheader("🌡️ 토픽 유사도 히트맵")
            components.html(
                f"<html><head><meta charset='utf-8'></head><body>{res['heatmap_html']}</body></html>",
                height=600, scrolling=True
            )

        # 계층 구조
        if res.get("hierarchy_html"):
            st.subheader("🌳 토픽 계층 구조")
            components.html(
                f"<html><head><meta charset='utf-8'></head><body>{res['hierarchy_html']}</body></html>",
                height=600, scrolling=True
            )
