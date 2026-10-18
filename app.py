import os
import glob
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

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
# Palet warna (dipakai bersama oleh CSS dan grafik agar tampilan seragam)
# ----------------------------------------------------------------------------
INK = "#0E2A47"
MUTED = "#5B6B7F"
INDIGO = "#3B6FD8"
TEAL = "#0EA5A4"
AMBER = "#F59E0B"
ROSE = "#E11D74"
VIOLET = "#7C3AED"
SKY = "#0284C7"
CLUSTER_COLORS = ["#3B6FD8", "#1E3A8A", "#60A5FA", "#94A3B8", "#0EA5A4", "#7C3AED", "#65A30D", "#EA580C", "#0F766E", "#9333EA"]

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, .stApp, .stMarkdown, label, p, h1, h2, h3,
    .stButton button, input, textarea, [data-baseweb="select"] {{
        font-family: 'Inter', system-ui, sans-serif;
    }}
    .stApp {{ background: #E9EDF3; color: {INK}; }}
    [data-testid="stHeader"] {{ background: transparent; }}
    .block-container {{
        max-width: 1200px; padding: 0 0 1.5rem 0; margin-top: 1.5rem;
        background: #F7F9FC; border: 1px solid #1F2937; border-radius: 4px;
    }}
    .stTabs {{ padding: 0 24px; }}

    /* Judul aplikasi */
    .app-header {{
        background: #ffffff; border-bottom: 1px solid #D9DEE5; padding: 14px 24px;
        font-size: 1.2rem; font-weight: 700; color: #1F2937; border-radius: 4px 4px 0 0;
    }}

    /* Tab */
    .stTabs [data-baseweb="tab-list"] {{ gap: 6px; border-bottom: none; }}
    .stTabs [data-baseweb="tab"] {{
        background: #EEF2F8; border-radius: 4px; padding: 6px 14px; height: auto;
        color: #4B5563; font-size: 0.85rem; font-weight: 600;
    }}
    .stTabs [aria-selected="true"] {{ background: #DCE8FB; color: #3B6FD8; }}
    .stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {{ display: none; }}

    /* Panel pengaturan */
    [data-testid="stVerticalBlockBorderWrapper"] {{
        background: #EEF2F8; border: 1px solid #D9DEE5; border-radius: 6px; box-shadow: none;
    }}
    .panel-title {{ font-size: 1rem; font-weight: 700; color: #1F2937; margin-bottom: 2px; }}
    .panel-sub {{ font-size: 0.8rem; color: {MUTED}; margin-bottom: 8px; }}

        /* Kolom isian putih seperti sketsa */
    [data-baseweb="select"] > div, .stNumberInput [data-baseweb="input"],
    .stNumberInput [data-baseweb="base-input"] {{ background: #ffffff; border-radius: 4px; }}

    /* Judul bagian */
    .section-title {{ font-size: 1rem; font-weight: 700; color: #1F2937; margin: 16px 0 8px 0; }}

    /* Kartu metrik */
    .mc {{
        background: #ffffff; border: 1px solid #D9DEE5; border-radius: 6px;
        padding: 10px 14px; min-height: 66px;
    }}
    .mc-l {{ font-size: 0.78rem; font-weight: 600; color: #374151; margin-bottom: 4px; }}
    .mc-v {{ font-size: 1.3rem; font-weight: 700; color: #1B8A3A; line-height: 1.1; }}

    /* Status kosong */
    .empty {{
        border: 1px dashed #C9D2E6; border-radius: 6px; padding: 28px 20px;
        text-align: center; color: {MUTED}; background: #ffffff; margin-top: 16px;
    }}
    .empty b {{ color: #1F2937; }}

    /* Tombol */
    div.stButton > button, div.stDownloadButton > button {{
        width: 100%; border-radius: 4px; font-weight: 600; padding: 0.55rem 1rem;
    }}
    div.stButton > button {{ background: #3B6FD8; color: #fff; border: none; }}
    div.stButton > button:hover {{ background: #2F5BB8; color: #fff; }}
    div.stDownloadButton > button {{ background: #fff; color: {INK}; border: 1px solid #D9DEE5; }}
    div.stDownloadButton > button:hover {{ background: #F1F4F9; color: {INK}; }}
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


def metric_card(label, value, color=None, tint=None):
    st.markdown(
        f'<div class="mc"><div class="mc-l">{label}</div><div class="mc-v">{value}</div></div>',
        unsafe_allow_html=True,
    )


def section_title(text):
    st.markdown(f'<div class="section-title">{text}</div>', unsafe_allow_html=True)


def panel_title(title, sub):
    st.markdown(
        f'<div class="panel-title">{title}</div><div class="panel-sub">{sub}</div>',
        unsafe_allow_html=True,
    )


def empty_state(title, text):
    st.markdown(
        f'<div class="empty"><b>{title}</b><br>{text}</div>',
        unsafe_allow_html=True,
    )


def style_ax(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#C9D2E6")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.grid(color="#E6EAF4", linewidth=0.8)
    ax.set_axisbelow(True)


def new_fig(w=6.4, h=4.2):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    style_ax(ax)
    return fig, ax


tab_supervised, tab_unsupervised = st.tabs(["Supervised Learning", "Unsupervised Learning"])

# ============================================================================
# SUPERVISED LEARNING
# ============================================================================
with tab_supervised:
    left, right = st.columns([1, 2.7], gap="large")

    with left:
        with st.container(border=True):
            panel_title("Pengaturan", "Pilih data, target, dan algoritma.")
            sup_upload = st.file_uploader(
                "Dataset (CSV)", type=["csv"], key="sup_upload",
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
                target = st.selectbox("Kolom target", all_cols, index=default_idx)

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

                test_size = st.slider("Ukuran data uji (%)", 10, 40, 20, 5)

                if task == "Regresi":
                    algo_options = ["Random Forest", "Linear Regression"]
                else:
                    algo_options = ["Random Forest", "Logistic Regression"]
                algorithm = st.selectbox("Algoritma", algo_options)

                max_depth = None
                if algorithm == "Random Forest":
                    max_depth = st.number_input("max_depth", min_value=1, max_value=50, value=5, step=1)
                random_seed = st.number_input("Random seed", 0, 9999, 42, 1)

                train_clicked = st.button("Mulai Training", key="train_supervised")
            else:
                train_clicked = False

    with right:
        if customer_df is None:
            empty_state(
                "Dataset pelanggan belum ditemukan",
                "Unggah file CSV di panel Pengaturan, atau letakkan "
                "<code>customer_satisfaction_data (1).csv</code> satu folder dengan app.py.",
            )
        else:
            section_title("Cuplikan Dataset")
            st.dataframe(customer_df.head(6), use_container_width=True, hide_index=True)
            st.caption(
                f"{customer_df.shape[0]:,} baris • {customer_df.shape[1]} kolom • "
                f"{int(customer_df.isna().sum().sum())} nilai kosong"
            )

            if not train_clicked:
                empty_state(
                    "Siap melatih model",
                    "Atur parameter di panel kiri, lalu klik <b>Mulai Training</b> untuk melihat hasil evaluasi.",
                )
            elif not selected_features:
                st.error("Pilih setidaknya satu fitur prediksi.")
            else:
                data = customer_df[selected_features + [target]].copy()
                if task == "Regresi":
                    data[target] = pd.to_numeric(data[target], errors="coerce")
                data = data.dropna(subset=[target])

                if task == "Klasifikasi":
                    # Target numerik dengan banyak nilai dibulatkan agar menjadi kelas.
                    if pd.api.types.is_numeric_dtype(data[target]):
                        if data[target].nunique() > 15:
                            data[target] = data[target].round().astype(int)
                        elif (data[target] % 1 == 0).all():
                            data[target] = data[target].astype(int)
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
                            metric_card("MAE", f"{mae:.3f}", INDIGO, "#EEF0FF")
                        with c2:
                            metric_card("RMSE", f"{rmse:.3f}", TEAL, "#E3F6F5")
                        with c3:
                            metric_card("R²", f"{r2:.3f}", ROSE, "#FDE7F1")
                        st.caption(
                            "MAE dan RMSE lebih kecil umumnya lebih baik. R² mendekati 1 berarti "
                            "model menjelaskan variasi target lebih baik; nilai negatif berarti "
                            "performa buruk pada data uji."
                        )

                        section_title("Aktual vs Prediksi")
                        fig, ax = new_fig()
                        ax.scatter(y_test, predictions, alpha=0.7, color=INDIGO, s=26, edgecolors="white", linewidths=0.5)
                        low = min(float(np.min(y_test)), float(np.min(predictions)))
                        high = max(float(np.max(y_test)), float(np.max(predictions)))
                        ax.plot([low, high], [low, high], linestyle="--", color=AMBER, linewidth=2, label="Prediksi sempurna")
                        ax.set_xlabel("Nilai aktual")
                        ax.set_ylabel("Nilai prediksi")
                        ax.legend(frameon=False, labelcolor=MUTED)
                        st.pyplot(fig)
                        plt.close(fig)
                    else:
                        acc = accuracy_score(y_test, predictions)
                        prec = precision_score(y_test, predictions, average="weighted", zero_division=0)
                        rec = recall_score(y_test, predictions, average="weighted", zero_division=0)
                        f1 = f1_score(y_test, predictions, average="weighted", zero_division=0)
                        c1, c2, c3, c4 = st.columns(4)
                        with c1:
                            metric_card("Accuracy", f"{acc:.2f}", INDIGO, "#EEF0FF")
                        with c2:
                            metric_card("Precision", f"{prec:.2f}", TEAL, "#E3F6F5")
                        with c3:
                            metric_card("Recall", f"{rec:.2f}", "#B45309", "#FEF3D7")
                        with c4:
                            metric_card("F1-Measure", f"{f1:.2f}", ROSE, "#FDE7F1")
                        st.caption("Precision, Recall, dan F1 dihitung dengan rata-rata tertimbang antar kelas.")

                        section_title("Confusion Matrix")
                        labels_sorted = sorted(y.unique(), key=lambda v: (len(v), v))
                        cm = confusion_matrix(y_test, predictions, labels=labels_sorted)
                        cmap = LinearSegmentedColormap.from_list("greys", ["#F3F4F6", "#6B7280"])
                        fig, ax = new_fig(5.2, 4.2)
                        ax.grid(False)
                        ax.imshow(cm, cmap=cmap)
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
                                    color="white" if cm[i, j] > thresh else INK,
                                    fontweight="bold",
                                )
                        st.pyplot(fig)
                        plt.close(fig)
                        st.caption("Baris = kelas aktual, kolom = kelas hasil prediksi.")

                    results = pd.DataFrame({
                        "Nilai aktual": y_test.to_numpy(),
                        "Hasil prediksi": predictions,
                    })
                    with st.expander("Lihat tabel hasil prediksi"):
                        st.dataframe(results.head(30), use_container_width=True, hide_index=True)
                    st.download_button(
                        "Unduh hasil prediksi (CSV)",
                        data=results.to_csv(index=False).encode("utf-8"),
                        file_name="hasil_prediksi.csv",
                        mime="text/csv",
                    )

# ============================================================================
# UNSUPERVISED LEARNING
# ============================================================================
with tab_unsupervised:
    left, right = st.columns([1, 2.7], gap="large")

    with left:
        with st.container(border=True):
            panel_title("Pengaturan", "Pilih kolom teks dan jumlah kelompok.")
            news_upload = st.file_uploader(
                "Dataset (CSV)", type=["csv"], key="news_upload",
                help="Opsional jika file CSV sudah ada di repositori GitHub.",
            )
            news_df, news_name = load_dataset(
                news_upload, UNSUPERVISED_CANDIDATES, "dataset_berita*.csv"
            )

            news_ready = False
            cluster_clicked = False
            if news_df is not None:
                st.caption(f"📄 {news_name}")

                text_cols = [
                    c for c in news_df.columns
                    if not pd.api.types.is_numeric_dtype(news_df[c])
                ]
                if not text_cols:
                    st.error("Tidak ada kolom teks pada dataset.")
                else:
                    news_ready = True
                    text_col = st.selectbox(
                        "Kolom teks yang dianalisis",
                        text_cols,
                        index=text_cols.index("text") if "text" in text_cols else 0,
                    )
                    st.selectbox("Algoritma", ["K-Means"])
                    n_clusters = st.slider("Jumlah cluster (K)", 2, 10, 4, 1)
                    lo_docs = min(200, max(1, len(news_df)))
                    hi_docs = max(min(3000, len(news_df)), lo_docs + 1)
                    max_docs = st.slider(
                        "Jumlah berita diproses",
                        min_value=lo_docs,
                        max_value=hi_docs,
                        value=min(1500, hi_docs),
                        step=100 if len(news_df) >= 300 else 10,
                    )
                    max_features = st.select_slider(
                        "Batas fitur TF-IDF", options=[1000, 2000, 3000, 5000], value=2000
                    )
                    seed = st.number_input("Random seed", 0, 9999, 42, 1, key="news_seed")
                    cluster_clicked = st.button("Mulai Clustering", key="run_clustering")

    with right:
        if news_df is None:
            empty_state(
                "Dataset berita belum ditemukan",
                "Unggah <code>dataset_berita_cluster.csv</code> di panel Pengaturan, "
                "atau letakkan file itu satu folder dengan app.py.",
            )
        elif news_ready:
            section_title("Cuplikan Dataset")
            preview = news_df.head(5).copy()
            preview[text_col] = preview[text_col].astype(str).str.slice(0, 160) + "…"
            st.dataframe(preview, use_container_width=True, hide_index=True)
            st.caption(f"{news_df.shape[0]:,} baris • {news_df.shape[1]} kolom")

            if not cluster_clicked:
                empty_state(
                    "Siap mengelompokkan berita",
                    "Atur jumlah cluster di panel kiri, lalu klik <b>Mulai Clustering</b>.",
                )
            else:
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

                        has_cat = "category_id" in work.columns
                        k1, k2, k3 = st.columns(3)
                        with k1:
                            metric_card("Silhouette Score", sil_text, INDIGO, "#EEF0FF")
                        with k2:
                            if has_cat:
                                try:
                                    ari = adjusted_rand_score(work["category_id"], labels)
                                    metric_card("Adjusted Rand Index", f"{ari:.2f}", TEAL, "#E3F6F5")
                                except Exception:
                                    metric_card("Adjusted Rand Index", "-", TEAL, "#E3F6F5")
                            else:
                                metric_card("Jumlah cluster", str(int(n_clusters)), TEAL, "#E3F6F5")
                        with k3:
                            metric_card("Berita diproses", f"{len(work):,}", "#B45309", "#FEF3D7")
                        st.caption(
                            "Silhouette lebih tinggi umumnya berarti cluster lebih terpisah. "
                            "category_id tidak dipakai saat membentuk cluster; hanya pembanding (ARI)."
                        )

                        g1, g2 = st.columns([1.7, 1], gap="large")
                        with g1:
                            st.markdown("**Sebaran cluster (proyeksi 2D)**")
                            try:
                                svd = TruncatedSVD(n_components=2, random_state=int(seed))
                                projection = svd.fit_transform(matrix)
                                centers = svd.transform(kmeans.cluster_centers_)
                                fig, ax = new_fig(6.6, 4.6)
                                for cid in range(int(n_clusters)):
                                    mask = labels == cid
                                    ax.scatter(
                                        projection[mask, 0], projection[mask, 1],
                                        color=CLUSTER_COLORS[cid % len(CLUSTER_COLORS)],
                                        alpha=0.7, s=22, edgecolors="white", linewidths=0.4,
                                        label=f"Cluster {cid}",
                                    )
                                ax.scatter(
                                    centers[:, 0], centers[:, 1], marker="o",
                                    facecolors="none", edgecolors=AMBER, s=150, linewidths=2.2,
                                    label="Centroid", zorder=5,
                                )
                                ax.set_xlabel("Komponen 1")
                                ax.set_ylabel("Komponen 2")
                                ax.legend(frameon=False, labelcolor=MUTED, fontsize=8, ncol=2)
                                st.pyplot(fig)
                                plt.close(fig)
                            except Exception:
                                st.info("Visualisasi 2D tidak tersedia untuk konfigurasi ini.")
                        with g2:
                            st.markdown("**Jumlah berita per cluster**")
                            counts = work["cluster"].value_counts().sort_index()
                            fig, ax = new_fig(4, 4.6)
                            ax.barh(
                                [f"Cluster {i}" for i in counts.index], counts.values,
                                color=[CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i in counts.index],
                            )
                            ax.invert_yaxis()
                            ax.grid(axis="y", visible=False)
                            st.pyplot(fig)
                            plt.close(fig)

                        section_title("Contoh berita per cluster")
                        for cluster_id in sorted(work["cluster"].unique()):
                            n_in = int((work["cluster"] == cluster_id).sum())
                            with st.expander(f"Cluster {cluster_id} — {n_in} berita"):
                                examples = work.loc[work["cluster"] == cluster_id, text_col].head(3)
                                for i, article in enumerate(examples, start=1):
                                    st.markdown(
                                        f"**Contoh {i}:** {article[:500]}{'…' if len(article) > 500 else ''}"
                                    )

                        st.download_button(
                            "Unduh hasil clustering (CSV)",
                            data=work.to_csv(index=False).encode("utf-8"),
                            file_name="hasil_clustering_berita.csv",
                            mime="text/csv",
                        )
                    except ValueError as error:
                        st.error(f"Clustering tidak dapat dijalankan: {error}")
                        st.info("Coba kurangi jumlah cluster atau batas fitur TF-IDF.")

st.caption("Proyek 3 — Implementasi Antarmuka Machine Learning")
