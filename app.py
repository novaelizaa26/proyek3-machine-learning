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
st.markdown("""
<style>
.block-container {padding-top:1.3rem; max-width:1400px}
div[data-testid="stMetric"] {background:white;border:1px solid #e5e7eb;padding:12px;border-radius:10px}
</style>
""", unsafe_allow_html=True)
st.title("🧪 Machine Learning Experiment")
st.caption("Supervised Learning untuk klasifikasi dan Unsupervised Learning untuk pengelompokan pelanggan.")
mode = st.radio("Pilih eksperimen", ["Supervised Learning", "Unsupervised Learning", "Tentang"], horizontal=True)

def repo_csv(names):
    for name in names:
        p = ROOT / name
        if p.exists():
            try:
                return pd.read_csv(p), name
            except Exception:
                pass
    return None, None

def preview(df, n=5):
    st.dataframe(df.head(n), use_container_width=True, hide_index=True)

if mode == "Supervised Learning":
    st.header("Supervised Learning")
    st.subheader("Klasifikasi Data")
    st.write("Pilih kolom target berlabel. Random Forest akan dievaluasi dengan Accuracy, Precision, Recall, F1-Score, dan Confusion Matrix.")
    base, base_name = repo_csv(["iris.csv", "Iris.csv", "customer_satisfaction_data (1).csv", "customer_satisfaction_data.csv"])
    upload = st.file_uploader("Unggah dataset klasifikasi (CSV)", type="csv", key="sup_upload")
    if upload:
        try:
            df, data_name = pd.read_csv(upload), upload.name
        except Exception as e:
            st.error(f"CSV tidak dapat dibaca: {e}")
            st.stop()
    else:
        df, data_name = base, base_name

    if df is None:
        st.warning("Unggah dataset klasifikasi CSV. Contoh paling sederhana adalah Iris dengan kolom target species.")
        st.info("File customer satisfaction yang sudah ada bisa dicoba jika memiliki kolom kategori sebagai target. Jika targetnya angka kontinu, gunakan dataset klasifikasi seperti Iris.")
    else:
        df.columns = [str(c).strip() for c in df.columns]
        st.success(f"Dataset dimuat: {data_name} — {len(df):,} baris dan {len(df.columns)} kolom.")
        with st.expander("Lihat contoh data"):
            preview(df, 8)
        cols = list(df.columns)
        preferred = next((i for i,c in enumerate(cols) if c.lower() in ["species","class","target","label","satisfaction level","satisfaction_level","satisfaction category"]), 0)
        left, right = st.columns([1, 2], gap="large")
        with left:
            st.markdown("#### Pengaturan")
            target = st.selectbox("Target column", cols, index=preferred)
            test_pct = st.slider("Test size (%)", 10, 40, 20, 5)
            trees = st.slider("Jumlah pohon", 50, 300, 100, 50)
            depth_choice = st.selectbox("max_depth", ["None", "3", "5", "10", "20"], index=2)
            run = st.button("🚀 Mulai Training", type="primary")
        with right:
            st.markdown("#### Cuplikan Dataset")
            preview(df, 5)

        if run:
            work = df.dropna(subset=[target]).copy()
            y = work[target].astype(str)
            if len(work) < 10 or y.nunique() < 2:
                st.error("Dibutuhkan setidaknya 10 baris dan minimal 2 kelas target.")
                st.stop()
            if y.nunique() > max(20, int(len(work)*0.5)):
                st.error("Target memiliki terlalu banyak nilai unik. Pilih kolom kategori/kelas, bukan ID atau nilai kontinu.")
                st.stop()
            X = work.drop(columns=[target]).dropna(axis=1, how="all")
            if X.empty:
                st.error("Tidak ada kolom fitur selain target.")
                st.stop()
            num = X.select_dtypes(include=np.number).columns.tolist()
            cat = [c for c in X.columns if c not in num]
            transformers = []
            if num:
                transformers.append(("numeric", Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]), num))
            if cat:
                transformers.append(("categorical", Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]), cat))
            prep = ColumnTransformer(transformers=transformers)
            depth = None if depth_choice == "None" else int(depth_choice)
            model = Pipeline([
                ("preprocess", prep),
                ("classifier", RandomForestClassifier(n_estimators=trees, max_depth=depth, random_state=42, class_weight="balanced")),
            ])
            stratify = y if y.value_counts().min() >= 2 else None
            try:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_pct/100, random_state=42, stratify=stratify)
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
            except Exception as e:
                st.error(f"Training gagal: {e}")
                st.stop()

            st.divider()
            st.subheader("Hasil Evaluasi")
            a,b,c,d = st.columns(4)
            a.metric("Accuracy", f"{accuracy_score(y_test,pred):.3f}")
            b.metric("Precision", f"{precision_score(y_test,pred,average='weighted',zero_division=0):.3f}")
            c.metric("Recall", f"{recall_score(y_test,pred,average='weighted',zero_division=0):.3f}")
            d.metric("F1-Measure", f"{f1_score(y_test,pred,average='weighted',zero_division=0):.3f}")
            labels = sorted(pd.Series(y_test).unique().tolist())
            cm = confusion_matrix(y_test, pred, labels=labels)
            fig, ax = plt.subplots(figsize=(6,4))
            ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels).plot(ax=ax, cmap="Blues", colorbar=False, xticks_rotation=35)
            ax.set_title("Confusion Matrix")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            with st.expander("Laporan klasifikasi lengkap"):
                st.dataframe(pd.DataFrame(classification_report(y_test,pred,zero_division=0,output_dict=True)).transpose(), use_container_width=True)
            result = pd.DataFrame({"Nilai aktual": y_test.reset_index(drop=True), "Hasil prediksi": pd.Series(pred)})
            st.download_button("Unduh hasil prediksi CSV", result.to_csv(index=False).encode("utf-8"), "hasil_supervised.csv", "text/csv")

elif mode == "Unsupervised Learning":
    st.header("Unsupervised Learning")
    st.subheader("Clustering Pelanggan dengan K-Means")
    st.write("Pilih setidaknya dua fitur numerik pelanggan. Model mengelompokkan data tanpa membutuhkan kolom target.")
    base, base_name = repo_csv(["Mall_Customers.csv", "mall_customers.csv", "customer_data.csv", "customer_satisfaction_data (1).csv", "customer_satisfaction_data.csv"])
    upload = st.file_uploader("Unggah dataset pelanggan (CSV)", type="csv", key="cluster_upload")
    if upload:
        try:
            df, data_name = pd.read_csv(upload), upload.name
        except Exception as e:
            st.error(f"CSV tidak dapat dibaca: {e}")
            st.stop()
    else:
        df, data_name = base, base_name

    if df is None:
        st.warning("Unggah dataset pelanggan dengan minimal dua kolom numerik.")
        st.info("Dataset tambahan yang disarankan: Mall_Customers.csv, biasanya memiliki CustomerID, Gender, Age, Annual Income (k$), dan Spending Score (1-100).")
    else:
        df.columns = [str(c).strip() for c in df.columns]
        st.success(f"Dataset dimuat: {data_name} — {len(df):,} baris dan {len(df.columns)} kolom.")
        with st.expander("Lihat contoh data"):
            preview(df, 8)
        numeric = df.select_dtypes(include=np.number).columns.tolist()
        features_default = [c for c in numeric if c.lower() not in ["id","customerid","customer_id","unnamed: 0"]]
        if len(features_default) < 2:
            st.warning("Dataset harus mempunyai minimal dua kolom numerik yang dapat dianalisis.")
        else:
            left, right = st.columns([1,2], gap="large")
            with left:
                st.markdown("#### Pengaturan")
                features = st.multiselect("Fitur yang dianalisis", features_default, default=features_default[:2])
                st.text_input("Algorithm", value="K-Means", disabled=True)
                k = st.slider("Jumlah cluster (K)", 2, 10, 3)
                run = st.button("🚀 Mulai Clustering", type="primary")
            with right:
                st.markdown("#### Cuplikan Dataset")
                preview(df, 5)
            if run:
                if len(features) < 2:
                    st.error("Pilih minimal dua fitur numerik.")
                    st.stop()
                work = df[features].replace([np.inf,-np.inf], np.nan).dropna()
                if len(work) <= k:
                    st.error(f"Jumlah baris valid harus lebih besar dari K={k}.")
                    st.stop()
                scaled = StandardScaler().fit_transform(work)
                try:
                    labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(scaled)
                except Exception as e:
                    st.error(f"Clustering gagal: {e}")
                    st.stop()
                score = silhouette_score(scaled, labels) if len(set(labels)) > 1 and len(set(labels)) < len(labels) else None
                st.divider()
                st.subheader("Hasil Clustering")
                a,b,c = st.columns(3)
                a.metric("Data dianalisis", f"{len(work):,}")
                b.metric("Jumlah cluster", str(len(set(labels))))
                c.metric("Silhouette Score", f"{score:.3f}" if score is not None else "N/A")
                result = work.copy()
                result["Cluster"] = labels
                fig, ax = plt.subplots(figsize=(8,4.5))
                ax.scatter(work[features[0]], work[features[1]], c=labels, cmap="tab10", alpha=.8, edgecolors="white", linewidths=.4)
                ax.set_xlabel(features[0]); ax.set_ylabel(features[1]); ax.set_title("Hasil K-Means Clustering"); ax.grid(alpha=.2)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
                st.markdown("#### Ringkasan tiap cluster")
                st.dataframe(result.groupby("Cluster")[features].mean().round(2), use_container_width=True)
                st.markdown("#### Data dengan label cluster")
                st.dataframe(result.head(100), use_container_width=True, hide_index=True)
                st.download_button("Unduh hasil clustering CSV", result.to_csv(index=False).encode("utf-8"), "hasil_clustering_pelanggan.csv", "text/csv")

else:
    st.header("Tentang aplikasi")
    st.markdown("""
**Supervised Learning**
- Algoritma: Random Forest Classifier
- Memerlukan kolom target berlabel
- Evaluasi: Accuracy, Precision, Recall, F1-Score, dan Confusion Matrix

**Unsupervised Learning**
- Algoritma: K-Means
- Tidak memerlukan kolom target
- Evaluasi: Silhouette Score, ringkasan cluster, dan scatter plot

Nilai evaluasi bergantung pada dataset dan pembagian data; hasil tidak harus sama persis dengan mockup.
""")
