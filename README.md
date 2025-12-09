# Human Activity Recognition: Comparative Analysis of Feature Selection Methods
### (İnsan Aktivitesi Tanıma: Öznitelik Seçimi Yöntemlerinin Karşılaştırmalı Analizi)

![Status](https://img.shields.io/badge/Status-Work_in_Progress-yellow) ![Python](https://img.shields.io/badge/Python-3.x-blue) ![Library](https://img.shields.io/badge/Library-Scikit--Learn-orange)

##  English Description

### Project Overview
This project aims to analyze the impact of different **Feature Selection** methods on model performance, computational efficiency, and interpretability. Using the **Human Activity Recognition (HAR) Using Smartphones** dataset, we classify human activities (walking, sitting, laying, etc.) based on sensor data.

The primary goal is to reduce the dimensionality of the dataset (originally **561 features**) while maintaining high accuracy and preventing **Overfitting**.

### Dataset
* **Source:** [Human Activity Recognition Using Smartphones](https://www.kaggle.com/datasets/uciml/human-activity-recognition-with-smartphones)
* **Instances:** 10,299 (Train + Test)
* **Features:** 561 (derived from Accelerometer and Gyroscope raw signals)
* **Classes:** 6 (Walking, Walking Upstairs, Walking Downstairs, Sitting, Standing, Laying)

### Methodology & Workflow
1.  **Exploratory Data Analysis (EDA) & Preprocessing:**
    * Checked for missing values and class imbalance.
    * Detected and removed duplicate column names.
    * Encoded target variables using Label Encoding.
    * **Constraint:** Subject IDs were removed to prevent data leakage and ensure the model learns generalized patterns.

2.  **Baseline Model:**
    * A Random Forest Classifier was trained on all 561 features to establish a performance benchmark.

3.  **Feature Selection Methods Applied:**
    * **Filter Method:** ANOVA F-value (`SelectKBest`).
    * **Embedded Method:** Random Forest Feature Importance (`SelectFromModel`).
    * **Wrapper Method:** Recursive Feature Elimination (RFE) with Decision Tree estimator.

4.  **Model Evaluation:**
    * Models are evaluated using **5-Fold Cross Validation**.
    * Metrics: Accuracy, F1-Score, Training Time.

### Preliminary Results (Current Status)
* **Baseline (561 Features):** ~92.6% Accuracy.
* **Wrapper Method (RFE - 100 Features):** ~91.3% Accuracy (Highest efficiency/performance trade-off).
* **Filter Method (ANOVA - 100 Features):** ~88.7% Accuracy.

*Note: The detailed technical report and final interpretation of the selected sensors are currently being written.*

---

## 🇹🇷 Türkçe Açıklama

### Proje Özeti
Bu proje, makine öğrenmesi modellerinde farklı **Feature Selection** (Öznitelik Seçimi) yöntemlerinin model performansı, hesaplama maliyeti ve yorumlanabilirlik üzerindeki etkilerini analiz etmeyi amaçlamaktadır. Projede **Human Activity Recognition (HAR)** veri seti kullanılarak sensör verileri üzerinden insan aktiviteleri (yürüme, oturma, yatma vb.) sınıflandırılmaktadır.

Temel hedef, 561 öznitelikten oluşan yüksek boyutlu veri setini indirgeyerek, doğruluktan ödün vermeden daha verimli ve **Overfitting** riskinden uzak bir model oluşturmaktır.

### Veri Seti
* **Kaynak:** [Human Activity Recognition Using Smartphones](https://www.kaggle.com/datasets/uciml/human-activity-recognition-with-smartphones)
* **Gözlem Sayısı:** 10,299 (Eğitim + Test)
* **Öznitelikler:** 561 adet (İvmeölçer ve Jiroskop verilerinden türetilmiş)
* **Sınıflar:** 6 Farklı Aktivite

### Yöntem ve Akış
1.  **Exploratory Data Analysis (EDA) & Preprocessing:**
    * Eksik veri ve sınıf dengesizliği kontrol edildi.
    * Tekrar eden (duplicate) sütun isimleri temizlendi.
    * Target değişkenler Label Encoding ile sayısallaştırıldı.
    * **Önemli:** Modelin kişiye özel ezber yapmasını (Data Leakage) önlemek için "Subject" verisi eğitimden çıkarıldı.

2.  **Baseline Model:**
    * Karşılaştırma yapabilmek için tüm (561) öznitelikler kullanılarak bir Random Forest modeli eğitildi.

3.  **Uygulanan Feature Selection Yöntemleri:**
    * **Filter Method:** ANOVA F-test istatistiği (`SelectKBest` kullanılarak).
    * **Embedded Method:** Random Forest Feature Importance (`SelectFromModel` kullanılarak).
    * **Wrapper Method:** Recursive Feature Elimination (RFE).

4.  **Model Değerlendirme:**
    * Tüm modeller **5-Fold Cross Validation** ile test edilmiştir.
    * Metrikler: Accuracy, F1-Score ve Eğitim Süresi (Training Time).

### Güncel Sonuçlar (Ön İzleme)
Şu ana kadar yapılan analizlerde:
* **Baseline Model:** %92.6 doğruluk oranına ulaştı.
* **Wrapper Method (RFE):** Özniteliklerin %82'si atılmasına rağmen %91.3 doğruluk oranı ile en iyi performansı gösterdi.
* **Filter Method:** En hızlı yöntem olmasına rağmen doğruluk oranı %88.7 seviyesinde kaldı.

*Not: Projenin detaylı teknik raporu ve seçilen sensörlerin fiziksel yorumlaması (Domain Knowledge) üzerindeki çalışmalar devam etmektedir.*

---

### Installation & Usage (Kurulum ve Kullanım)

# Dataset Setup (Veri Seti Kurulumu)
⚠️ **Note:** Due to license and size constraints, the dataset is not included in this repository.
⚠️ **Not:** Lisans ve boyut kısıtlamaları nedeniyle veri seti bu repoya dahil edilmemiştir.

1.  Download the dataset from Kaggle: [Human Activity Recognition with Smartphones](https://www.kaggle.com/uciml/human-activity-recognition-with-smartphones)
2.  Extract `train.csv` and `test.csv` files into the `data/` folder.
    *(İndirdiğiniz csv dosyalarını `data/` klasörünün içine atın.)*

# Project Structure (Proje Yapısı)
```text
repo-name/
├── data/                 # Place train.csv and test.csv here (Veri dosyaları buraya)
├── notebooks/            # Jupyter Notebooks are here (Kod dosyaları buraya)
│   └── Proje_Notebook.ipynb
├── requirements.txt      # Dependencies (Gerekli kütüphaneler)
└── README.md