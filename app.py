import os
import glob
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
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    silhouette_score,
    adjusted_rand_score,
)
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Machine Learning Experiment",
    page_icon="🤖",
    layout="wide",
)

# ----------------------------------------------------------------------------
# Gaya tampilan (mengikuti sketsa antarmuka)
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem; max-width: 1250px;}
    .app-header {
        border: 1px solid #d9dee5; border-radius: 6px 6px 0 0;
        padding: 12px 18px; background: #ffffff;
        font-size: 1.35rem; font-weight: 700; color: #1f2937;
        margin-bottom: 0.6rem;
    }
    .panel-title {font-size: 1.05rem; font-weight: 700; color: #1f2937; margin-bottom: 4px;}
    .section-title {font-size: 1.05rem; font-weight: 700; color: #1f2937; margin: 14px 0 6px 0;}
    .metric-card {
        border: 1px solid #d9dee5; border-radius: 6px; padding: 8px 14px;
        background: #ffffff; min-height: 64px;
    }
    .metric-label {font-size: 0.78rem; color: #4b5563; margin-bottom: 2px;}
    .metric-value {font-size: 1.25rem; font-weight: 700; color: #1b8a3a;}
    .stTabs [data-baseweb="tab-list"] {gap: 6px;}
    .stTabs [data-baseweb="tab"] {
        background: #eef2f7; border-radius: 6px; padding: 6px 16px; height: auto;
    }
    .stTabs [aria-selected="true"] {background: #dbeafe; color: #1d4ed8; font-weight: 600;}
    div.stButton > button {
        width: 100%; background: #3b6fd8; color: white; border: none; font-weight: 600;
    }
    div.stButton > button:hover {background: #2f5bb8; color: white;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="app-header">Machine Learning Experiment</div>', unsafe_allow_html=True)

SUPERVISED_CANDIDATES = [
    "customer_satisfaction_data (1).csv",
    "customer_satisfaction_data 1.csv",
    "customer_satisfaction_data.csv",
]
UNSUPERVISED_CANDIDATES = ["dataset_berita_cluster.csv"]


# ----------------------------------------------------------------------------
# Fungsi bantu
# ----------------------------------------------------------------------------
def find_default_file(candidates, pattern):
    for path in candidates:
        if os.path.exists(path):
            return path
    matches = glob.glob(pattern)
    return matches[0] if matches else None


@st.cache_data(show_spinner=False)
def load_csv_path(path):
    return pd.read_csv(path)


def load_dataset(uploaded_file, candidates, pattern):
    """Kembalikan (dataframe, nama_file) dari unggahan atau file di repositori."""
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file), uploaded_file.name
    path = find_default_file(candidates, pattern)
    if path:
        return load_csv_path(path), os.path.basename(path)
    return None, None


def make_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def metric_card(label, value):
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div></div>',
        unsafe_allow_html=True,
    )


def section_title(text):
    st.markdown(f'<div class="section-title">{text}</div>', unsafe_allow_html=True)


def panel_title(text):
    st.markdown(f'<div class="panel-title">{text}</div>', unsafe_allow_html=True)


tab_supervised, tab_unsupervised = st.tabs(["Supervised Learning", "Unsupervised Learning"])

# ============================================================================
# SUPERVISED LEARNING
# ============================================================================
with tab_supervised:
    left, right = st.columns([1, 3], gap="large")

    with left:
        with st.container(border=True):
            panel_title("Pengaturan")
            sup_upload = st.file_uploader(
                "Dataset", type=["csv"], key="sup_upload",
                help="Opsional jika file CSV sudah ada di repositori GitHub.",
            )
            customer_df, customer_name = load_dataset(
                sup_upload, SUPERVISED_CANDIDATES, "customer_satisfaction*.csv"
            )

            if customer_df is not None:
                st.caption(f"📄 {customer_name}")

                numeric_cols = [
                    c for c in customer_df.columns
                    if pd.api.types.is_numeric_dtype(customer_df[c])
                ]
                all_cols = customer_df.columns.tolist()
                default_idx = (
                    all_cols.index("Satisfaction Score")
                    if "Satisfaction Score" in all_cols else 0
                )
                target = st.selectbox("Target column", all_cols, index=default_idx)

                target_is_numeric = target in numeric_cols
                task_options = ["Regresi", "Klasifikasi"] if target_is_numeric else ["Klasifikasi"]
                task = st.selectbox("Jenis tugas", task_options)

                exclude_id = st.checkbox("Keluarkan kolom ID", value=True)
                feature_candidates = [c for c in all_cols if c != target]
                if exclude_id:
                    feature_candidates = [
                        c for c in feature_candidates
                        if c.strip().lower() not in {"customer id", "customer_id", "id"}
                    ]
                selected_features = st.multiselect(
                    "Fitur prediksi", feature_candidates, default=feature_candidates
                )

                test_size = st.slider("Test size (%)", 10, 40, 20, 5)

                if task == "Regresi":
                    algo_options = ["Random Forest", "Linear Regression"]
                else:
                    algo_options = ["Random Forest", "Logistic Regression"]
                algorithm = st.selectbox("Algorithm", algo_options)

                max_depth = None
                if algorithm == "Random Forest":
                    max_depth = st.number_input("max_depth", min_value=1, max_value=50, value=5, step=1)
                random_seed = st.number_input("Random seed", 0, 9999, 42, 1)

                train_clicked = st.button("Mulai Training", key="train_supervised")
            else:
                train_clicked = False

    with right:
        if customer_df is None:
            st.warning(
                "Dataset pelanggan belum ditemukan. Unggah file CSV di panel Pengaturan, "
                "atau letakkan `customer_satisfaction_data (1).csv` satu folder dengan app.py."
            )
        else:
            section_title("Cuplikan Dataset")
            st.dataframe(customer_df.head(5), use_container_width=True, hide_index=True)
            st.caption(
                f"{customer_df.shape[0]:,} baris • {customer_df.shape[1]} kolom • "
                f"{int(customer_df.isna().sum().sum())} nilai kosong"
            )

            if train_clicked:
                if not selected_features:
                    st.error("Pilih setidaknya satu fitur prediksi.")
                else:
                    data = customer_df[selected_features + [target]].copy()
                    if task == "Regresi":
                        data[target] = pd.to_numeric(data[target], errors="coerce")
                    data = data.dropna(subset=[target])

                    if task == "Klasifikasi":
                        # Target numerik dengan banyak nilai dibulatkan agar menjadi kelas.
                        if pd.api.types.is_numeric_dtype(data[target]) and data[target].nunique() > 15:
                            data[target] = data[target].round().astype(int)
                        data[target] = data[target].astype(str)

                    if len(data) < 10:
                        st.error("Data berlabel terlalu sedikit untuk dibagi menjadi data latih dan uji.")
                    else:
                        X = data[selected_features]
                        y = data[target]

                        numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
                        categorical_features = [c for c in X.columns if c not in numeric_features]

                        transformers = []
                        if numeric_features:
                            transformers.append((
                                "num",
                                Pipeline([
                                    ("imputer", SimpleImputer(strategy="median")),
                                    ("scaler", StandardScaler()),
                                ]),
                                numeric_features,
                            ))
                        if categorical_features:
                            transformers.append((
                                "cat",
                                Pipeline([
                                    ("imputer", SimpleImputer(strategy="most_frequent")),
                                    ("onehot", make_one_hot_encoder()),
                                ]),
                                categorical_features,
                            ))
                        preprocess = ColumnTransformer(transformers=transformers)

                        seed_i = int(random_seed)
                        if task == "Regresi":
                            if algorithm == "Random Forest":
                                estimator = RandomForestRegressor(
                                    n_estimators=150, max_depth=int(max_depth),
                                    random_state=seed_i, n_jobs=-1,
                                )
                            else:
                                estimator = LinearRegression()
                        else:
                            if algorithm == "Random Forest":
                                estimator = RandomForestClassifier(
                                    n_estimators=150, max_depth=int(max_depth),
                                    random_state=seed_i, n_jobs=-1,
                                )
                            else:
                                estimator = LogisticRegression(max_iter=1000)

                        model = Pipeline([("preprocess", preprocess), ("model", estimator)])

                        try:
                            strat = y if task == "Klasifikasi" else None
                            X_train, X_test, y_train, y_test = train_test_split(
                                X, y, test_size=test_size / 100, random_state=seed_i, stratify=strat
                            )
                        except ValueError:
                            X_train, X_test, y_train, y_test = train_test_split(
                                X, y, test_size=test_size / 100, random_state=seed_i
                            )

                        with st.spinner("Melatih model..."):
                            model.fit(X_train, y_train)
                            predictions = model.predict(X_test)

                        section_title("Hasil Evaluasi")

                        if task == "Regresi":
                            mae = mean_absolute_error(y_test, predictions)
                            rmse = np.sqrt(mean_squared_error(y_test, predictions))
                            r2 = r2_score(y_test, predictions)
                            c1, c2, c3 = st.columns(3)
                            with c1:
                                metric_card("MAE", f"{mae:.3f}")
                            with c2:
                                metric_card("RMSE", f"{rmse:.3f}")
                            with c3:
                                metric_card("R²", f"{r2:.3f}")
                            st.caption(
                                "MAE dan RMSE lebih kecil umumnya lebih baik. R² mendekati 1 berarti "
                                "model menjelaskan variasi target lebih baik; nilai negatif berarti "
                                "performa buruk pada data uji."
                            )

                            section_title("Aktual vs Prediksi")
                            fig, ax = plt.subplots(figsize=(6, 4))
                            ax.scatter(y_test, predictions, alpha=0.7, color="#3b6fd8")
                            low = min(float(np.min(y_test)), float(np.min(predictions)))
                            high = max(float(np.max(y_test)), float(np.max(predictions)))
                            ax.plot([low, high], [low, high], linestyle="--", color="gray")
                            ax.set_xlabel("Nilai aktual")
                            ax.set_ylabel("Nilai prediksi")
                            st.pyplot(fig)
                            plt.close(fig)

                            results = pd.DataFrame({
                                "Nilai aktual": y_test.to_numpy(),
                                "Hasil prediksi": predictions,
                            })
                        else:
                            acc = accuracy_score(y_test, predictions)
                            prec = precision_score(y_test, predictions, average="weighted", zero_division=0)
                            rec = recall_score(y_test, predictions, average="weighted", zero_division=0)
                            f1 = f1_score(y_test, predictions, average="weighted", zero_division=0)
                            c1, c2, c3, c4 = st.columns(4)
                            with c1:
                                metric_card("Accuracy", f"{acc:.2f}")
                            with c2:
                                metric_card("Precision", f"{prec:.2f}")
                            with c3:
                                metric_card("Recall", f"{rec:.2f}")
                            with c4:
                                metric_card("F1-Measure", f"{f1:.2f}")
                            st.caption("Precision, Recall, dan F1 dihitung dengan rata-rata tertimbang antar kelas.")

                            section_title("Confusion Matrix")
                            labels_sorted = sorted(y.unique(), key=lambda v: (len(v), v))
                            cm = confusion_matrix(y_test, predictions, labels=labels_sorted)
                            fig, ax = plt.subplots(figsize=(5, 4))
                            ax.imshow(cm, cmap="Greys")
                            ax.set_xticks(range(len(labels_sorted)))
                            ax.set_yticks(range(len(labels_sorted)))
                            ax.set_xticklabels(labels_sorted)
                            ax.set_yticklabels(labels_sorted)
                            ax.set_xlabel("Prediksi")
                            ax.set_ylabel("Aktual")
                            thresh = cm.max() / 2 if cm.max() > 0 else 0
                            for i in range(cm.shape[0]):
                                for j in range(cm.shape[1]):
                                    ax.text(
                                        j, i, int(cm[i, j]), ha="center", va="center",
                                        color="white" if cm[i, j] > thresh else "black",
                                        fontweight="bold",
                                    )
                            st.pyplot(fig)
                            plt.close(fig)
                            st.caption("Visualisasi hasil prediksi model")

                            results = pd.DataFrame({
                                "Nilai aktual": y_test.to_numpy(),
                                "Hasil prediksi": predictions,
                            })

                        with st.expander("Lihat tabel hasil prediksi"):
                            st.dataframe(results.head(30), use_container_width=True, hide_index=True)
                        st.download_button(
                            "Unduh hasil prediksi CSV",
                            data=results.to_csv(index=False).encode("utf-8"),
                            file_name="hasil_prediksi.csv",
                            mime="text/csv",
                        )

# ============================================================================
# UNSUPERVISED LEARNING
# ============================================================================
with tab_unsupervised:
    left, right = st.columns([1, 3], gap="large")

    with left:
        with st.container(border=True):
            panel_title("Pengaturan")
            news_upload = st.file_uploader(
                "Dataset", type=["csv"], key="news_upload",
                help="Opsional jika file CSV sudah ada di repositori GitHub.",
            )
            news_df, news_name = load_dataset(
                news_upload, UNSUPERVISED_CANDIDATES, "dataset_berita*.csv"
            )

            if news_df is not None:
                st.caption(f"📄 {news_name}")

                text_cols = [
                    c for c in news_df.columns
                    if not pd.api.types.is_numeric_dtype(news_df[c])
                ]
                if not text_cols:
                    st.error("Tidak ada kolom teks pada dataset.")
                    news_ready = False
                else:
                    news_ready = True
                    text_col = st.selectbox(
                        "Fitur yang dianalisis (teks)",
                        text_cols,
                        index=text_cols.index("text") if "text" in text_cols else 0,
                    )
                    algo_cluster = st.selectbox("Algorithm", ["K-Means"])
                    n_clusters = st.slider("Jumlah cluster (K)", 2, 10, 4, 1)
                    max_docs = st.slider(
                        "Jumlah berita diproses",
                        min_value=min(200, max(1, len(news_df))),
                        max_value=max(min(3000, len(news_df)), min(200, max(1, len(news_df))) + 1),
                        value=min(1500, len(news_df)),
                        step=100 if len(news_df) >= 300 else 10,
                    )
                    max_features = st.select_slider(
                        "Batas fitur TF-IDF", options=[1000, 2000, 3000, 5000], value=2000
                    )
                    seed = st.number_input("Random seed", 0, 9999, 42, 1, key="news_seed")
                    cluster_clicked = st.button("Mulai Clustering", key="run_clustering")
            else:
                news_ready = False
                cluster_clicked = False

    with right:
        if news_df is None:
            st.warning(
                "Dataset berita belum ditemukan. Unggah `dataset_berita_cluster.csv` di panel "
                "Pengaturan, atau letakkan file itu satu folder dengan app.py."
            )
        elif news_ready:
            section_title("Cuplikan Dataset")
            preview = news_df.head(5).copy()
            preview[text_col] = preview[text_col].astype(str).str.slice(0, 160) + "…"
            st.dataframe(preview, use_container_width=True, hide_index=True)
            st.caption(f"{news_df.shape[0]:,} baris • {news_df.shape[1]} kolom")

            if cluster_clicked:
                keep_cols = [text_col] + (["category_id"] if "category_id" in news_df.columns else [])
                work = news_df[keep_cols].copy()
                work[text_col] = work[text_col].fillna("").astype(str).str.strip()
                work = work[work[text_col].str.len() > 0].head(int(max_docs)).reset_index(drop=True)

                if len(work) <= n_clusters:
                    st.error("Jumlah teks valid harus lebih banyak daripada jumlah cluster.")
                else:
                    try:
                        with st.spinner("Mengubah teks menjadi TF-IDF dan menjalankan K-Means..."):
                            vectorizer = TfidfVectorizer(
                                stop_words="english",
                                max_features=int(max_features),
                                min_df=2,
                                max_df=0.95,
                            )
                            matrix = vectorizer.fit_transform(work[text_col])
                            kmeans = KMeans(
                                n_clusters=int(n_clusters), random_state=int(seed), n_init=10
                            )
                            labels = kmeans.fit_predict(matrix)
                            work["cluster"] = labels

                        section_title("Hasil Clustering")
                        res_left, res_right = st.columns([1, 2], gap="large")

                        with res_left:
                            sil_text = "-"
                            try:
                                sil_sample = min(1000, matrix.shape[0])
                                if len(set(labels)) > 1 and sil_sample > n_clusters:
                                    sil = silhouette_score(
                                        matrix, labels, sample_size=sil_sample,
                                        random_state=int(seed),
                                    )
                                    sil_text = f"{sil:.2f}"
                            except Exception:
                                pass
                            metric_card("Silhouette Score", sil_text)

                            if "category_id" in work.columns:
                                try:
                                    ari = adjusted_rand_score(work["category_id"], labels)
                                    st.write("")
                                    metric_card("Adjusted Rand Index (vs category_id)", f"{ari:.2f}")
                                except Exception:
                                    pass

                            st.write("")
                            st.caption(
                                "Silhouette lebih tinggi umumnya berarti cluster lebih terpisah. "
                                "category_id tidak dipakai saat membentuk cluster; hanya pembanding."
                            )
                            counts = work["cluster"].value_counts().sort_index()
                            st.bar_chart(counts.rename("Jumlah berita"))

                        with res_right:
                            st.markdown("**Scatter Plot**")
                            try:
                                svd = TruncatedSVD(n_components=2, random_state=int(seed))
                                projection = svd.fit_transform(matrix)
                                centers = svd.transform(kmeans.cluster_centers_)
                                fig, ax = plt.subplots(figsize=(6, 4.2))
                                ax.scatter(
                                    projection[:, 0], projection[:, 1],
                                    c=labels, cmap="tab10", alpha=0.65, s=18,
                                )
                                ax.scatter(
                                    centers[:, 0], centers[:, 1],
                                    facecolors="none", edgecolors="orange",
                                    s=140, linewidths=2, label="Centroid",
                                )
                                ax.set_xlabel("Fitur 1")
                                ax.set_ylabel("Fitur 2")
                                ax.legend()
                                st.pyplot(fig)
                                plt.close(fig)
                            except Exception:
                                st.info("Visualisasi 2D tidak tersedia untuk konfigurasi ini.")

                        st.markdown("**Contoh berita per cluster**")
                        for cluster_id in sorted(work["cluster"].unique()):
                            n_in = int((work["cluster"] == cluster_id).sum())
                            with st.expander(f"Cluster {cluster_id} — {n_in} berita"):
                                examples = work.loc[work["cluster"] == cluster_id, text_col].head(3)
                                for i, article in enumerate(examples, start=1):
                                    st.markdown(
                                        f"**Contoh {i}:** {article[:500]}{'…' if len(article) > 500 else ''}"
                                    )

                        st.download_button(
                            "Unduh hasil clustering CSV",
                            data=work.to_csv(index=False).encode("utf-8"),
                            file_name="hasil_clustering_berita.csv",
                            mime="text/csv",
                        )
                    except ValueError as error:
                        st.error(f"Clustering tidak dapat dijalankan: {error}")
                        st.info("Coba kurangi jumlah cluster atau batas fitur TF-IDF.")

st.caption("Proyek 3 — Implementasi Antarmuka Machine Learning")
