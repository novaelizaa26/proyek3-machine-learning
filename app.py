import os
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    silhouette_score,
    adjusted_rand_score,
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Machine Learning Experiment",
    page_icon="🤖",
    layout="wide",
)

# ---------------------------------------------------------------- Gaya tampilan
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem; max-width: 1200px;}
    .app-title {
        font-size: 1.4rem; font-weight: 700;
        border-bottom: 1px solid rgba(128,128,128,.35);
        padding-bottom: .6rem; margin-bottom: .6rem;
    }
    .panel-title {font-size: 1.05rem; font-weight: 700; margin-bottom: .2rem;}
    .section-title {font-size: 1.05rem; font-weight: 700; margin: .9rem 0 .4rem 0;}
    .metric-card {
        border: 1px solid rgba(128,128,128,.35);
        border-radius: 8px; padding: 10px 14px;
        background: var(--secondary-background-color);
    }
    .metric-label {font-size: .75rem; opacity: .75;}
    .metric-value {font-size: 1.45rem; font-weight: 700; color: #1a7f37;}
    div.stButton > button[kind="primary"] {
        width: 100%; font-weight: 600; border-radius: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="app-title">Machine Learning Experiment</div>', unsafe_allow_html=True)

# Dataset dapat diletakkan di folder yang sama dengan app.py pada repositori GitHub.
SUPERVISED_PATH = "customer_satisfaction_data (1).csv"
UNSUPERVISED_PATH = "dataset_berita_cluster.csv"


# ---------------------------------------------------------------- Fungsi bantu
def read_csv_safely(uploaded_file, default_path):
    """Baca file unggahan; jika tidak ada, coba baca dataset di repositori."""
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file), uploaded_file.name
    if os.path.exists(default_path):
        return pd.read_csv(default_path), default_path
    return None, None


def make_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def metric_card(container, label, value):
    container.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section(title):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)


tab_sup, tab_unsup = st.tabs(["Supervised Learning", "Unsupervised Learning"])

# ================================================================ SUPERVISED
with tab_sup:
    left, right = st.columns([1, 3], gap="medium")

    with left:
        with st.container(border=True):
            st.markdown('<div class="panel-title">Pengaturan</div>', unsafe_allow_html=True)
            sup_upload = st.file_uploader(
                "Dataset", type=["csv"], key="sup_upload",
                help="Opsional jika file sudah ada di repositori GitHub.",
            )
            customer_df, sup_name = read_csv_safely(sup_upload, SUPERVISED_PATH)

            if customer_df is not None:
                st.caption(f"📄 {sup_name}")
                target_options = [
                    c for c in customer_df.columns
                    if pd.api.types.is_numeric_dtype(customer_df[c])
                ]
                if target_options:
                    default_idx = (
                        target_options.index("Satisfaction Score")
                        if "Satisfaction Score" in target_options else 0
                    )
                    target = st.selectbox("Target column", target_options, index=default_idx)

                    id_names = {"customer id", "customer_id", "id"}
                    feature_candidates = [
                        c for c in customer_df.columns
                        if c != target and c.strip().lower() not in id_names
                    ]
                    with st.expander("Fitur prediksi (lanjutan)"):
                        selected_features = st.multiselect(
                            "Pilih fitur", feature_candidates, default=feature_candidates,
                            label_visibility="collapsed",
                        )

                    test_size = st.slider("Test size (%)", 10, 40, 20, step=5)
                    model_choice = st.selectbox(
                        "Algorithm", ["Random Forest Regressor", "Linear Regression"]
                    )
                    is_rf = model_choice == "Random Forest Regressor"
                    max_depth = st.number_input(
                        "max_depth", min_value=1, max_value=50, value=5, step=1,
                        disabled=not is_rf,
                    )
                    random_seed = st.number_input("Random seed", 0, 9999, 42, step=1)
                    run_sup = st.button("Mulai Training", type="primary", key="run_sup")
                else:
                    st.error("Tidak ada kolom numerik untuk dijadikan target regresi.")
                    run_sup = False
            else:
                st.warning(
                    "Dataset belum ditemukan. Unggah `customer_satisfaction_data (1).csv` "
                    "atau taruh di folder yang sama dengan app.py."
                )
                run_sup = False

    with right:
        if customer_df is None:
            st.info("Unggah dataset pelanggan di panel Pengaturan untuk memulai.")
        elif not target_options:
            pass
        else:
            section("Cuplikan Dataset")
            st.dataframe(customer_df.head(5), use_container_width=True, hide_index=True)
            st.caption(f"{customer_df.shape[0]:,} baris • {customer_df.shape[1]} kolom")

            if run_sup:
                if not selected_features:
                    st.error("Pilih setidaknya satu fitur prediksi.")
                else:
                    data = customer_df[selected_features + [target]].copy()
                    data[target] = pd.to_numeric(data[target], errors="coerce")
                    data = data.dropna(subset=[target])

                    if len(data) < 10:
                        st.error("Data berlabel terlalu sedikit untuk dibagi menjadi data latih dan uji.")
                    else:
                        with st.spinner("Melatih model..."):
                            X = data[selected_features]
                            y = data[target]
                            num_cols = X.select_dtypes(include=["number"]).columns.tolist()
                            cat_cols = [c for c in X.columns if c not in num_cols]

                            transformers = []
                            if num_cols:
                                transformers.append((
                                    "num",
                                    Pipeline([
                                        ("imputer", SimpleImputer(strategy="median")),
                                        ("scaler", StandardScaler()),
                                    ]),
                                    num_cols,
                                ))
                            if cat_cols:
                                transformers.append((
                                    "cat",
                                    Pipeline([
                                        ("imputer", SimpleImputer(strategy="most_frequent")),
                                        ("onehot", make_one_hot_encoder()),
                                    ]),
                                    cat_cols,
                                ))
                            preprocess = ColumnTransformer(transformers=transformers)

                            if is_rf:
                                estimator = RandomForestRegressor(
                                    n_estimators=150, max_depth=int(max_depth),
                                    random_state=int(random_seed), n_jobs=-1,
                                )
                            else:
                                estimator = LinearRegression()

                            model = Pipeline([("preprocess", preprocess), ("model", estimator)])
                            X_train, X_test, y_train, y_test = train_test_split(
                                X, y, test_size=test_size / 100, random_state=int(random_seed)
                            )
                            model.fit(X_train, y_train)
                            pred = model.predict(X_test)

                        st.session_state["sup_result"] = {
                            "mae": mean_absolute_error(y_test, pred),
                            "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
                            "r2": r2_score(y_test, pred),
                            "y_test": y_test.to_numpy(),
                            "pred": pred,
                            "target": target,
                            "model": model_choice,
                        }

            res = st.session_state.get("sup_result")
            if res:
                section("Hasil Evaluasi")
                c1, c2, c3, c4 = st.columns(4)
                metric_card(c1, "MAE", f"{res['mae']:.3f}")
                metric_card(c2, "RMSE", f"{res['rmse']:.3f}")
                metric_card(c3, "R²", f"{res['r2']:.3f}")
                metric_card(c4, "Algoritma", res["model"].replace(" Regressor", ""))
                st.caption(
                    "MAE/RMSE lebih kecil = lebih baik. R² mendekati 1 berarti model "
                    "menjelaskan variasi target dengan baik; nilai negatif berarti performa buruk."
                )

                section("Aktual vs Prediksi")
                fig, ax = plt.subplots(figsize=(6, 3.8))
                ax.scatter(res["y_test"], res["pred"], alpha=0.7, color="#1f6feb", s=22)
                lo = min(float(np.min(res["y_test"])), float(np.min(res["pred"])))
                hi = max(float(np.max(res["y_test"])), float(np.max(res["pred"])))
                ax.plot([lo, hi], [lo, hi], linestyle="--", color="gray")
                ax.set_xlabel("Nilai aktual")
                ax.set_ylabel("Nilai prediksi")
                ax.set_title(f"Prediksi {res['target']}")
                st.pyplot(fig)
                plt.close(fig)

                results_df = pd.DataFrame({
                    "Nilai aktual": res["y_test"], "Hasil prediksi": res["pred"]
                })
                with st.expander("Lihat tabel hasil prediksi"):
                    st.dataframe(results_df.head(30), use_container_width=True)
                st.download_button(
                    "Unduh hasil prediksi CSV",
                    data=results_df.to_csv(index=False).encode("utf-8"),
                    file_name="hasil_prediksi_kepuasan.csv",
                    mime="text/csv",
                )

# ================================================================ UNSUPERVISED
with tab_unsup:
    left, right = st.columns([1, 3], gap="medium")

    with left:
        with st.container(border=True):
            st.markdown('<div class="panel-title">Pengaturan</div>', unsafe_allow_html=True)
            news_upload = st.file_uploader(
                "Dataset", type=["csv"], key="news_upload",
                help="Opsional jika file sudah ada di repositori GitHub.",
            )
            news_df, news_name = read_csv_safely(news_upload, UNSUPERVISED_PATH)

            run_cluster = False
            if news_df is not None:
                st.caption(f"📄 {news_name}")
                text_options = [c for c in news_df.columns if news_df[c].dtype == object]
                if text_options:
                    text_col = st.selectbox(
                        "Fitur yang dianalisis (teks)",
                        text_options,
                        index=text_options.index("text") if "text" in text_options else 0,
                    )
                    st.selectbox("Algorithm", ["K-Means"])
                    n_clusters = st.slider("Jumlah cluster (K)", 2, 10, 4)

                    total = len(news_df)
                    if total > 200:
                        max_docs = st.slider(
                            "Jumlah berita diproses", 200, min(3000, total),
                            min(1500, total), step=100,
                        )
                    else:
                        max_docs = total
                    max_features = st.select_slider(
                        "Batas fitur TF-IDF", options=[1000, 2000, 3000, 5000], value=2000
                    )
                    seed = st.number_input("Random seed", 0, 9999, 42, step=1, key="news_seed")
                    run_cluster = st.button("Mulai Clustering", type="primary", key="run_cluster")
                else:
                    st.error("Tidak ditemukan kolom teks pada dataset.")
            else:
                st.warning(
                    "Dataset belum ditemukan. Unggah `dataset_berita_cluster.csv` "
                    "atau taruh di folder yang sama dengan app.py."
                )

    with right:
        if news_df is None:
            st.info("Unggah dataset berita di panel Pengaturan untuk memulai.")
        elif not text_options:
            pass
        else:
            section("Cuplikan Dataset")
            preview = news_df.head(5).copy()
            for c in preview.columns:
                if preview[c].dtype == object:
                    preview[c] = preview[c].astype(str).str.slice(0, 120)
            st.dataframe(preview, use_container_width=True, hide_index=True)
            st.caption(f"{news_df.shape[0]:,} baris • {news_df.shape[1]} kolom")

            if run_cluster:
                cols = [text_col] + (["category_id"] if "category_id" in news_df.columns else [])
                work = news_df[cols].copy().rename(columns={text_col: "text"})
                work["text"] = work["text"].fillna("").astype(str).str.strip()
                work = work[work["text"].str.len() > 0].head(int(max_docs)).reset_index(drop=True)

                if len(work) <= n_clusters:
                    st.error("Jumlah teks valid harus lebih banyak daripada jumlah cluster.")
                else:
                    try:
                        with st.spinner("Mengubah teks menjadi TF-IDF dan menjalankan K-Means..."):
                            vec = TfidfVectorizer(
                                stop_words="english", max_features=int(max_features),
                                min_df=2, max_df=0.95,
                            )
                            matrix = vec.fit_transform(work["text"])
                            km = KMeans(n_clusters=int(n_clusters), random_state=int(seed), n_init=10)
                            labels = km.fit_predict(matrix)
                            work["cluster"] = labels

                            sil = None
                            try:
                                sample = min(1000, matrix.shape[0])
                                if len(set(labels)) > 1 and sample > n_clusters:
                                    sil = silhouette_score(
                                        matrix, labels, sample_size=sample, random_state=int(seed)
                                    )
                            except Exception:
                                pass

                            ari = None
                            if "category_id" in work.columns:
                                try:
                                    ari = adjusted_rand_score(work["category_id"], labels)
                                except Exception:
                                    pass

                            proj = centers2d = None
                            if matrix.shape[1] >= 2 and matrix.shape[0] >= 3:
                                svd = TruncatedSVD(n_components=2, random_state=int(seed))
                                proj = svd.fit_transform(matrix)
                                centers2d = svd.transform(km.cluster_centers_)

                        st.session_state["unsup_result"] = {
                            "work": work, "labels": labels, "sil": sil, "ari": ari,
                            "proj": proj, "centers": centers2d, "k": int(n_clusters),
                        }
                    except ValueError as error:
                        st.error(f"Clustering tidak dapat dijalankan: {error}")
                        st.info("Coba kurangi jumlah cluster atau batas fitur TF-IDF.")

            res = st.session_state.get("unsup_result")
            if res:
                section("Hasil Clustering")
                c1, c2, c3 = st.columns(3)
                metric_card(c1, "Silhouette Score", f"{res['sil']:.3f}" if res["sil"] is not None else "-")
                metric_card(c2, "Jumlah cluster (K)", res["k"])
                metric_card(c3, "ARI vs category_id", f"{res['ari']:.3f}" if res["ari"] is not None else "-")
                st.caption(
                    "Skor Silhouette lebih tinggi umumnya berarti cluster lebih terpisah. "
                    "category_id (jika ada) tidak dipakai membentuk cluster, hanya sebagai pembanding."
                )

                g1, g2 = st.columns([3, 2])
                with g1:
                    section("Scatter Plot")
                    if res["proj"] is not None:
                        fig, ax = plt.subplots(figsize=(6, 4))
                        ax.scatter(
                            res["proj"][:, 0], res["proj"][:, 1],
                            c=res["labels"], cmap="tab10", alpha=0.65, s=18,
                        )
                        ax.scatter(
                            res["centers"][:, 0], res["centers"][:, 1],
                            facecolors="none", edgecolors="orange", s=120, linewidths=2,
                        )
                        ax.set_xlabel("Fitur 1")
                        ax.set_ylabel("Fitur 2")
                        st.pyplot(fig)
                        plt.close(fig)
                    else:
                        st.info("Visualisasi 2D tidak tersedia untuk konfigurasi ini.")
                with g2:
                    section("Jumlah berita per cluster")
                    counts = res["work"]["cluster"].value_counts().sort_index()
                    st.bar_chart(counts.rename("Jumlah berita"))

                section("Contoh berita per cluster")
                for cid in sorted(res["work"]["cluster"].unique()):
                    n = int((res["work"]["cluster"] == cid).sum())
                    with st.expander(f"Cluster {cid} — {n} berita"):
                        for i, art in enumerate(
                            res["work"].loc[res["work"]["cluster"] == cid, "text"].head(3), start=1
                        ):
                            st.markdown(f"**Contoh {i}:** {art[:500]}{'…' if len(art) > 500 else ''}")

                st.download_button(
                    "Unduh hasil clustering CSV",
                    data=res["work"].to_csv(index=False).encode("utf-8"),
                    file_name="hasil_clustering_berita.csv",
                    mime="text/csv",
                )

st.caption("Proyek 3 — Implementasi Antarmuka Machine Learning")
