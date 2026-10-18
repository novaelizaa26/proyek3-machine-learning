import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report,
    silhouette_score,
)
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent

st.set_page_config(page_title="Machine Learning Experiment", page_icon="🧪", layout="wide")

# Gaya antarmuka dibuat menyerupai mockup tugas: header tipis, panel pengaturan,
# area cuplikan dataset, kartu metrik, dan tombol aksi biru.
st.markdown("""
<style>
.block-container {padding: 1.1rem 1.6rem 2rem 1.6rem; max-width: 1500px;}
header[data-testid="stHeader"] {height: 2.3rem;}
h1 {font-size: 1.65rem !important; margin-bottom: .3rem !important;}
h2 {font-size: 1.18rem !important;}
h3 {font-size: 1rem !important;}
div[data-testid="stMetric"] {
  background: #ffffff; border: 1px solid #dce1e8; border-radius: 8px;
  padding: 10px 12px; min-height: 92px;
}
div[data-testid="stMetricLabel"] {font-size: .8rem;}
div[data-testid="stMetricValue"] {font-size: 1.45rem;}
div.stButton > button[kind="primary"] {
  background: #3674d9; border: 1px solid #3674d9; color: white;
  font-weight: 600; width: 100%; border-radius: 6px;
}
div.stButton > button[kind="primary"]:hover {background: #255fc0; border-color: #255fc0;}
section[data-testid="stSidebar"] {background: #f4f6f9;}
div[data-testid="stDataFrame"] {border: 1px solid #e1e5eb; border-radius: 5px;}
.small-note {color:#64748b; font-size:.82rem;}
</style>
""", unsafe_allow_html=True)

st.title("🧪 Machine Learning Experiment")
mode = st.radio(
    "Pilih eksperimen",
    ["Supervised Learning", "Unsupervised Learning", "Tentang"],
    horizontal=True,
    label_visibility="collapsed",
)

def find_csv(candidates):
    for name in candidates:
        path = ROOT / name
        if path.exists():
            try:
                return pd.read_csv(path), name
            except Exception:
                continue
    return None, None

def show_preview(df, rows=5):
    st.dataframe(df.head(rows), use_container_width=True, hide_index=True)

def load_user_data(upload, candidates, key):
    if upload is not None:
        try:
            return pd.read_csv(upload), upload.name
        except Exception as exc:
            st.error(f"File CSV tidak dapat dibaca: {exc}")
            st.stop()
    df, name = find_csv(candidates)
    if df is None:
        st.warning("Dataset belum ditemukan di repositori. Unggah file CSV milikmu melalui tombol di bawah.")
    return df, name

CUSTOMER_FILES = [
    "customer_satisfaction_data (1).csv",
    "customer_satisfaction_data.csv",
    "customer_satisfaction.csv",
]

if mode == "Supervised Learning":
    # Mockup: pengaturan kiri, cuplikan dataset dan hasil kanan.
    st.markdown("### Prediksi Kategori Kepuasan Pelanggan")
    st.caption("Latih model klasifikasi pada dataset pelanggan milikmu. Metrik dan confusion matrix ditampilkan setelah tombol training ditekan.")
    upload = st.file_uploader("Dataset pelanggan (CSV)", type=["csv"], key="supervised_upload")
    df, data_name = load_user_data(upload, CUSTOMER_FILES, "sup")

    if df is not None:
        df.columns = [str(c).strip() for c in df.columns]
        st.caption(f"Dataset: **{data_name}** · {len(df):,} baris · {len(df.columns)} kolom")
        left, right = st.columns([1, 2.35], gap="large")

        with left:
            st.markdown("#### Pengaturan")
            st.markdown("**Dataset**")
            st.text_input("Nama dataset", value=data_name, disabled=True, key="sup_dataset_name")
            cols = list(df.columns)
            # Jika kolom skor kepuasan berupa angka kontinu, jadikan kategori agar
            # eksperimen ini tetap merupakan supervised classification.
            target_guess = next(
                (c for c in cols if c.lower().replace("_", " ").strip() in
                 ["satisfaction score", "satisfaction level", "satisfaction", "satisfaction category", "label", "target"]),
                cols[-1],
            )
            target = st.selectbox("Target column", cols, index=cols.index(target_guess))
            test_pct = st.selectbox("Test size", [20, 25, 30], index=0, format_func=lambda x: f"{x}%")
            algorithm = st.selectbox("Algorithm", ["Random Forest"], index=0)
            max_depth = st.selectbox("max_depth", ["None", "3", "5", "10", "20"], index=2)
            run = st.button("🚀 Mulai Training", type="primary", key="run_supervised")

        with right:
            st.markdown("#### Cuplikan Dataset")
            show_preview(df, 5)
            st.markdown("#### Hasil Evaluasi")
            metrics_slot = st.empty()
            confusion_slot = st.empty()
            details_slot = st.empty()
            if not run:
                st.info("Atur target di panel kiri, lalu klik **Mulai Training** untuk melihat hasil evaluasi.")

        if run:
            work = df.dropna(subset=[target]).copy()
            raw_y = work[target]
            # Classification needs classes. Numeric continuous satisfaction scores
            # are converted to low / medium / high categories for model evaluation.
            if pd.api.types.is_numeric_dtype(raw_y) and raw_y.nunique() > 12:
                try:
                    work["_target_kategori_"] = pd.qcut(
                        raw_y, q=min(3, raw_y.nunique()), labels=["Rendah", "Sedang", "Tinggi"][:min(3, raw_y.nunique())],
                        duplicates="drop"
                    ).astype(str)
                    target_for_model = "_target_kategori_"
                    st.caption(f"Kolom **{target}** berupa angka dengan banyak nilai unik, jadi dikelompokkan menjadi kategori untuk klasifikasi.")
                except Exception:
                    work["_target_kategori_"] = raw_y.astype(str)
                    target_for_model = "_target_kategori_"
            else:
                target_for_model = target
                work[target_for_model] = work[target_for_model].astype(str)

            y = work[target_for_model].astype(str)
            X = work.drop(columns=[target, target_for_model], errors="ignore")
            # Exclude obvious ID columns from features.
            id_cols = [c for c in X.columns if c.lower().strip() in ["id", "customerid", "customer_id", "unnamed: 0"]]
            X = X.drop(columns=id_cols, errors="ignore")
            X = X.dropna(axis=1, how="all")
            if len(work) < 10 or y.nunique() < 2:
                st.error("Diperlukan setidaknya 10 baris dan minimal 2 kelas target.")
                st.stop()
            if y.value_counts().min() < 2:
                st.error("Setiap kelas target perlu memiliki setidaknya 2 data. Pilih target lain atau periksa dataset.")
                st.stop()
            if X.empty:
                st.error("Tidak ada kolom fitur yang bisa digunakan selain target.")
                st.stop()

            numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
            categorical_cols = [c for c in X.columns if c not in numeric_cols]
            transformers = []
            if numeric_cols:
                transformers.append(("numeric", Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                ]), numeric_cols))
            if categorical_cols:
                transformers.append(("categorical", Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]), categorical_cols))
            prep = ColumnTransformer(transformers=transformers)
            depth = None if max_depth == "None" else int(max_depth)
            model = Pipeline([
                ("preprocess", prep),
                ("classifier", RandomForestClassifier(
                    n_estimators=150, max_depth=depth, random_state=42,
                    class_weight="balanced"
                )),
            ])
            try:
                stratify = y if y.value_counts().min() >= 2 else None
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_pct / 100, random_state=42, stratify=stratify
                )
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
            except Exception as exc:
                st.error(f"Training gagal: {exc}")
                st.stop()

            with right:
                st.markdown("#### Hasil Evaluasi")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Accuracy", f"{accuracy_score(y_test, pred):.2f}")
                m2.metric("Precision", f"{precision_score(y_test, pred, average='weighted', zero_division=0):.2f}")
                m3.metric("Recall", f"{recall_score(y_test, pred, average='weighted', zero_division=0):.2f}")
                m4.metric("F1-Measure", f"{f1_score(y_test, pred, average='weighted', zero_division=0):.2f}")

                st.markdown("#### Confusion Matrix")
                labels = sorted(pd.Series(y_test).unique().tolist())
                cm = confusion_matrix(y_test, pred, labels=labels)
                fig, ax = plt.subplots(figsize=(5.4, 3.3))
                ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels).plot(
                    ax=ax, cmap="Blues", colorbar=False, xticks_rotation=30
                )
                ax.set_xlabel("Prediksi")
                ax.set_ylabel("Aktual")
                fig.tight_layout()
                st.pyplot(fig, use_container_width=False)
                plt.close(fig)
                st.caption("Visualisasi hasil model klasifikasi.")
                with st.expander("Lihat laporan klasifikasi"):
                    st.dataframe(
                        pd.DataFrame(classification_report(y_test, pred, zero_division=0, output_dict=True)).transpose(),
                        use_container_width=True,
                    )
                result = pd.DataFrame({
                    "Nilai aktual": y_test.reset_index(drop=True),
                    "Hasil prediksi": pd.Series(pred),
                })
                st.download_button(
                    "Unduh hasil prediksi CSV",
                    result.to_csv(index=False).encode("utf-8"),
                    "hasil_prediksi_kepuasan.csv", "text/csv",
                )

elif mode == "Unsupervised Learning":
    st.markdown("### Segmentasi Pelanggan dengan K-Means")
    st.caption("Kelompokkan pelanggan berdasarkan fitur numerik dari dataset milikmu tanpa membutuhkan kolom target.")
    upload = st.file_uploader("Dataset pelanggan (CSV)", type=["csv"], key="cluster_upload")
    df, data_name = load_user_data(upload, CUSTOMER_FILES, "cluster")

    if df is not None:
        df.columns = [str(c).strip() for c in df.columns]
        st.caption(f"Dataset: **{data_name}** · {len(df):,} baris · {len(df.columns)} kolom")
        numeric = df.select_dtypes(include=np.number).columns.tolist()
        numeric = [c for c in numeric if c.lower().strip() not in ["id", "customerid", "customer_id", "unnamed: 0"]]
        left, right = st.columns([1, 2.35], gap="large")

        with left:
            st.markdown("#### Pengaturan")
            st.markdown("**Dataset**")
            st.text_input("Nama dataset", value=data_name, disabled=True, key="cluster_dataset_name")
            default_features = numeric[:2]
            features = st.multiselect("Fitur yang dianalisis", numeric, default=default_features)
            st.text_input("Algorithm", value="K-Means", disabled=True, key="cluster_algorithm")
            k = st.number_input("Jumlah cluster (K)", min_value=2, max_value=10, value=3, step=1)
            run = st.button("🚀 Mulai Clustering", type="primary", key="run_clustering")

        with right:
            st.markdown("#### Cuplikan Dataset")
            show_preview(df, 5)
            if not run:
                st.markdown("#### Hasil Clustering")
                st.info("Pilih minimal dua fitur numerik, lalu klik **Mulai Clustering**.")
        if run:
            if len(features) < 2:
                st.error("Pilih setidaknya dua fitur numerik di panel kiri.")
                st.stop()
            work = df[features].replace([np.inf, -np.inf], np.nan).dropna()
            if len(work) <= k:
                st.error(f"Jumlah data valid harus lebih banyak daripada K={k}.")
                st.stop()
            scaler = StandardScaler()
            scaled = scaler.fit_transform(work)
            try:
                labels = KMeans(n_clusters=int(k), random_state=42, n_init=10).fit_predict(scaled)
            except Exception as exc:
                st.error(f"Clustering gagal: {exc}")
                st.stop()
            score = silhouette_score(scaled, labels) if 1 < len(set(labels)) < len(work) else None
            with right:
                st.markdown("#### Hasil Clustering")
                m1, m2 = st.columns(2)
                m1.metric("Jumlah Cluster", str(len(set(labels))))
                m2.metric("Silhouette Score", f"{score:.2f}" if score is not None else "N/A")
                st.markdown("#### Scatter Plot")
                fig, ax = plt.subplots(figsize=(7.2, 3.5))
                ax.scatter(work[features[0]], work[features[1]], c=labels, cmap="tab10", alpha=.85, edgecolors="white", linewidths=.45)
                ax.set_xlabel(features[0])
                ax.set_ylabel(features[1])
                ax.grid(alpha=.15)
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
                result = work.copy()
                result["Cluster"] = labels
                with st.expander("Ringkasan tiap cluster"):
                    st.dataframe(result.groupby("Cluster")[features].mean().round(2), use_container_width=True)
                with st.expander("Data dengan label cluster"):
                    st.dataframe(result.head(100), use_container_width=True, hide_index=True)
                st.download_button(
                    "Unduh hasil clustering CSV",
                    result.to_csv(index=False).encode("utf-8"),
                    "hasil_clustering_pelanggan.csv", "text/csv",
                )

else:
    st.markdown("### Tentang aplikasi")
    st.write("Aplikasi ini memakai dataset pelanggan milik pengguna untuk dua eksperimen machine learning.")
    st.markdown("""
- **Supervised Learning:** Random Forest Classifier, metrik Accuracy, Precision, Recall, F1-Measure, dan Confusion Matrix.
- **Unsupervised Learning:** K-Means, Silhouette Score, dan Scatter Plot.
- Dataset CSV dapat dibaca langsung dari repositori jika nama filenya tersedia, atau diunggah lewat aplikasi.
""")
