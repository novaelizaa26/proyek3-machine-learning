import os
import io
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
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Proyek 3 Machine Learning",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Proyek 3 — Aplikasi Machine Learning")
st.write(
    "Aplikasi ini menyediakan eksperimen **Supervised Learning** untuk prediksi "
    "kepuasan pelanggan dan **Unsupervised Learning** untuk pengelompokan teks berita."
)

# Dataset dapat diletakkan di folder yang sama dengan app.py pada repositori GitHub.
SUPERVISED_PATH = "customer_satisfaction_data (1).csv"
UNSUPERVISED_PATH = "dataset_berita_cluster.csv"


def read_csv_safely(uploaded_file, default_path):
    """Baca file unggahan; jika tidak ada, coba baca dataset di repositori."""
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    if os.path.exists(default_path):
        return pd.read_csv(default_path)
    return None


def make_one_hot_encoder():
    # Mendukung versi scikit-learn yang lebih baru maupun lama.
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


tab_supervised, tab_unsupervised, tab_about = st.tabs(
    ["📈 Supervised Learning", "📰 Unsupervised Learning", "ℹ️ Tentang"]
)

with tab_supervised:
    st.header("Prediksi Skor Kepuasan Pelanggan")
    st.write(
        "Dataset yang disiapkan menggunakan kolom **Satisfaction Score** sebagai "
        "target numerik. Karena itu, eksperimen utama menggunakan model regresi."
    )

    supervised_upload = st.file_uploader(
        "Unggah dataset pelanggan CSV (opsional jika file sudah ada di GitHub)",
        type=["csv"],
        key="supervised_upload",
    )
    customer_df = read_csv_safely(supervised_upload, SUPERVISED_PATH)

    if customer_df is None:
        st.warning(
            "Dataset pelanggan belum ditemukan. Unggah file "
            "`customer_satisfaction_data (1).csv` di atas, atau tambahkan file itu "
            "ke repositori GitHub pada folder yang sama dengan app.py."
        )
    else:
        st.success(f"Dataset dimuat: {customer_df.shape[0]:,} baris dan {customer_df.shape[1]} kolom.")
        with st.expander("Lihat contoh data"):
            st.dataframe(customer_df.head(10), use_container_width=True)
            st.write("Jumlah nilai kosong per kolom:")
            st.dataframe(customer_df.isna().sum().rename("Nilai kosong"))

        target_options = [
            c for c in customer_df.columns
            if pd.api.types.is_numeric_dtype(customer_df[c])
        ]
        default_target = (
            target_options.index("Satisfaction Score")
            if "Satisfaction Score" in target_options
            else 0
        )

        if not target_options:
            st.error("Tidak ditemukan kolom target numerik untuk regresi.")
        else:
            target = st.selectbox(
                "Pilih kolom target yang ingin diprediksi",
                target_options,
                index=default_target,
            )
            exclude_id = st.checkbox(
                "Keluarkan kolom ID pelanggan jika tersedia",
                value=True,
                help="Kolom ID biasanya hanya penanda, bukan fitur prediksi.",
            )
            feature_candidates = [c for c in customer_df.columns if c != target]
            if exclude_id:
                feature_candidates = [
                    c for c in feature_candidates
                    if c.strip().lower() not in {"customer id", "customer_id", "id"}
                ]
            selected_features = st.multiselect(
                "Pilih fitur prediksi",
                feature_candidates,
                default=feature_candidates,
            )

            model_choice = st.selectbox(
                "Pilih algoritma regresi",
                ["Random Forest Regressor", "Linear Regression"],
            )
            test_size = st.slider(
                "Persentase data uji (%)", min_value=10, max_value=40, value=20, step=5
            )
            random_seed = st.number_input(
                "Random seed", min_value=0, max_value=9999, value=42, step=1
            )

            if st.button("🚀 Latih dan evaluasi model", key="train_supervised"):
                if not selected_features:
                    st.error("Pilih setidaknya satu fitur prediksi.")
                else:
                    data = customer_df[selected_features + [target]].copy()
                    data[target] = pd.to_numeric(data[target], errors="coerce")
                    data = data.dropna(subset=[target])

                    if len(data) < 10:
                        st.error("Data berlabel yang tersedia terlalu sedikit untuk dibagi menjadi data latih dan uji.")
                    else:
                        X = data[selected_features]
                        y = data[target]
                        numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
                        categorical_features = [c for c in X.columns if c not in numeric_features]

                        numeric_pipeline = Pipeline(
                            steps=[
                                ("imputer", SimpleImputer(strategy="median")),
                                ("scaler", StandardScaler()),
                            ]
                        )
                        categorical_pipeline = Pipeline(
                            steps=[
                                ("imputer", SimpleImputer(strategy="most_frequent")),
                                ("onehot", make_one_hot_encoder()),
                            ]
                        )
                        transformers = []
                        if numeric_features:
                            transformers.append(("num", numeric_pipeline, numeric_features))
                        if categorical_features:
                            transformers.append(("cat", categorical_pipeline, categorical_features))

                        preprocess = ColumnTransformer(transformers=transformers)
                        if model_choice == "Random Forest Regressor":
                            estimator = RandomForestRegressor(
                                n_estimators=150,
                                random_state=int(random_seed),
                                n_jobs=-1,
                            )
                        else:
                            estimator = LinearRegression()

                        model = Pipeline(
                            steps=[("preprocess", preprocess), ("model", estimator)]
                        )
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=test_size / 100,
                            random_state=int(random_seed)
                        )
                        model.fit(X_train, y_train)
                        predictions = model.predict(X_test)

                        mae = mean_absolute_error(y_test, predictions)
                        rmse = np.sqrt(mean_squared_error(y_test, predictions))
                        r2 = r2_score(y_test, predictions)

                        st.subheader("Hasil evaluasi")
                        m1, m2, m3 = st.columns(3)
                        m1.metric("MAE", f"{mae:.3f}")
                        m2.metric("RMSE", f"{rmse:.3f}")
                        m3.metric("R²", f"{r2:.3f}")
                        st.caption(
                            "MAE dan RMSE yang lebih kecil umumnya lebih baik. "
                            "R² mendekati 1 menunjukkan model menjelaskan variasi target lebih baik; "
                            "nilai negatif bisa terjadi dan berarti performa buruk pada data uji."
                        )

                        results = pd.DataFrame({
                            "Nilai aktual": y_test.to_numpy(),
                            "Hasil prediksi": predictions,
                        })
                        st.write("Perbandingan nilai aktual dan prediksi")
                        st.dataframe(results.head(30), use_container_width=True)

                        fig, ax = plt.subplots()
                        ax.scatter(y_test, predictions, alpha=0.7)
                        low = min(float(np.min(y_test)), float(np.min(predictions)))
                        high = max(float(np.max(y_test)), float(np.max(predictions)))
                        ax.plot([low, high], [low, high], linestyle="--")
                        ax.set_xlabel("Nilai aktual")
                        ax.set_ylabel("Nilai prediksi")
                        ax.set_title("Aktual vs Prediksi")
                        st.pyplot(fig)
                        plt.close(fig)

                        st.download_button(
                            "Unduh hasil prediksi CSV",
                            data=results.to_csv(index=False).encode("utf-8"),
                            file_name="hasil_prediksi_kepuasan.csv",
                            mime="text/csv",
                        )

with tab_unsupervised:
    st.header("Clustering Teks Berita")
    st.write(
        "Teks berita diubah menjadi fitur TF-IDF, lalu dikelompokkan dengan K-Means. "
        "Kolom `category_id`, jika tersedia, tidak digunakan untuk membentuk cluster; "
        "kolom itu hanya opsional untuk membandingkan cluster dengan kategori asal."
    )

    news_upload = st.file_uploader(
        "Unggah dataset berita CSV (opsional jika file sudah ada di GitHub)",
        type=["csv"],
        key="news_upload",
    )
    news_df = read_csv_safely(news_upload, UNSUPERVISED_PATH)

    if news_df is None:
        st.warning(
            "Dataset berita belum ditemukan. Unggah `dataset_berita_cluster.csv` "
            "di atas, atau tambahkan file tersebut ke repositori GitHub."
        )
    else:
        st.success(f"Dataset dimuat: {news_df.shape[0]:,} baris dan {news_df.shape[1]} kolom.")
        if "text" not in news_df.columns:
            st.error("Kolom `text` tidak ditemukan. Periksa kembali dataset berita.")
        else:
            with st.expander("Lihat contoh teks berita"):
                st.dataframe(news_df.head(5), use_container_width=True)

            max_docs = st.slider(
                "Jumlah berita yang diproses",
                min_value=min(200, max(1, len(news_df))),
                max_value=min(3000, len(news_df)),
                value=min(1500, len(news_df)),
                step=100 if len(news_df) >= 300 else 10,
                help="Untuk mempercepat aplikasi, pemrosesan dibatasi. Naikkan jumlahnya jika diperlukan.",
            )
            n_clusters = st.slider(
                "Jumlah cluster (K)",
                min_value=2,
                max_value=10,
                value=4,
                step=1,
            )
            max_features = st.select_slider(
                "Batas fitur TF-IDF",
                options=[1000, 2000, 3000, 5000],
                value=2000,
            )
            seed = st.number_input(
                "Random seed clustering",
                min_value=0,
                max_value=9999,
                value=42,
                step=1,
                key="news_seed",
            )

            if st.button("🧩 Jalankan clustering berita", key="run_clustering"):
                work = news_df[["text"] + (["category_id"] if "category_id" in news_df.columns else [])].copy()
                work["text"] = work["text"].fillna("").astype(str).str.strip()
                work = work[work["text"].str.len() > 0].head(int(max_docs)).reset_index(drop=True)

                if len(work) <= n_clusters:
                    st.error("Jumlah teks valid harus lebih banyak daripada jumlah cluster.")
                else:
                    with st.spinner("Mengubah teks menjadi TF-IDF dan menjalankan K-Means..."):
                        vectorizer = TfidfVectorizer(
                            stop_words="english",
                            max_features=int(max_features),
                            min_df=2,
                            max_df=0.95,
                        )
                        try:
                            matrix = vectorizer.fit_transform(work["text"])
                            kmeans = KMeans(
                                n_clusters=int(n_clusters),
                                random_state=int(seed),
                                n_init=10,
                            )
                            labels = kmeans.fit_predict(matrix)
                            work["cluster"] = labels

                            st.subheader("Ringkasan cluster")
                            counts = work["cluster"].value_counts().sort_index()
                            st.bar_chart(counts.rename("Jumlah berita"))

                            try:
                                sil_sample = min(1000, matrix.shape[0])
                                if len(set(labels)) > 1 and sil_sample > n_clusters:
                                    score = silhouette_score(
                                        matrix, labels, sample_size=sil_sample,
                                        random_state=int(seed)
                                    )
                                    st.metric("Silhouette score (sampel)", f"{score:.3f}")
                                    st.caption("Skor lebih tinggi umumnya berarti cluster lebih terpisah; interpretasikan bersama contoh teks.")
                            except Exception:
                                st.info("Silhouette score tidak dapat dihitung untuk hasil ini.")

                            if "category_id" in work.columns:
                                try:
                                    ari = adjusted_rand_score(work["category_id"], labels)
                                    st.metric("Adjusted Rand Index vs category_id", f"{ari:.3f}")
                                    st.caption(
                                        "Ini hanya evaluasi pembanding menggunakan label kategori yang tersedia; "
                                        "category_id tidak dipakai saat membentuk cluster."
                                    )
                                except Exception:
                                    pass

                            st.write("Contoh berita per cluster")
                            for cluster_id in sorted(work["cluster"].unique()):
                                with st.expander(f"Cluster {cluster_id} — {int((work['cluster'] == cluster_id).sum())} berita"):
                                    examples = work.loc[work["cluster"] == cluster_id, "text"].head(3)
                                    for i, article in enumerate(examples, start=1):
                                        st.markdown(f"**Contoh {i}:** {article[:700]}{'…' if len(article) > 700 else ''}")

                            # Proyeksi 2D untuk visualisasi; tidak digunakan untuk membentuk cluster.
                            if matrix.shape[1] >= 2 and matrix.shape[0] >= 3:
                                try:
                                    projection = TruncatedSVD(n_components=2, random_state=int(seed)).fit_transform(matrix)
                                    fig, ax = plt.subplots()
                                    scatter = ax.scatter(
                                        projection[:, 0], projection[:, 1],
                                        c=labels, alpha=0.65, s=18
                                    )
                                    ax.set_title("Visualisasi cluster berita (proyeksi 2D)")
                                    ax.set_xlabel("Komponen 1")
                                    ax.set_ylabel("Komponen 2")
                                    st.pyplot(fig)
                                    plt.close(fig)
                                except Exception:
                                    st.info("Visualisasi 2D tidak tersedia untuk konfigurasi ini.")

                            st.download_button(
                                "Unduh hasil clustering CSV",
                                data=work.to_csv(index=False).encode("utf-8"),
                                file_name="hasil_clustering_berita.csv",
                                mime="text/csv",
                            )
                        except ValueError as error:
                            st.error(f"Clustering tidak dapat dijalankan: {error}")
                            st.info("Coba kurangi jumlah cluster atau batas fitur TF-IDF.")

with tab_about:
    st.header("Tentang aplikasi")
    st.markdown(
        """
        **Supervised Learning**
        - Target bawaan: `Satisfaction Score` (regresi).
        - Algoritma: Random Forest Regressor atau Linear Regression.
        - Evaluasi: MAE, RMSE, dan R².

        **Unsupervised Learning**
        - Fitur teks: kolom `text`.
        - Representasi teks: TF-IDF.
        - Algoritma: K-Means.
        - Evaluasi: Silhouette Score; jika `category_id` tersedia, ARI ditampilkan sebagai pembanding.

        Hasil bergantung pada dataset dan parameter yang dipilih. Evaluasi pada data uji bukan jaminan performa pada data baru.
        """
    )
    st.caption("Proyek 3 — Implementasi Antarmuka Machine Learning")
