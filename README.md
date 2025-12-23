# Comparative Analysis of Feature Selection in HAR 🏃‍♂️📱

*(Türkçe açıklama için aşağı kaydırınız / Scroll down for Turkish description)*

---

## 🇬🇧 English Description

### 📌 Project Overview
This project performs a comparative analysis of different **Feature Selection methods** (Filter, Embedded, Wrapper) using the **UCI Human Activity Recognition (HAR)** dataset.

The main objective is to reduce the high-dimensional feature space (561 features) while maintaining high classification accuracy for human activities such as Walking, Sitting, and Standing.

### 🛠 Tech Stack
* **Language:** Python 3.x
* **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
* **Techniques:** * ANOVA F-Value (Filter)
  * Random Forest Importance (Embedded)
  * Recursive Feature Elimination - RFE (Wrapper)

### 📊 Key Results
The analysis showed that reducing the feature count from **561 to ~100** resulted in minimal accuracy loss while significantly reducing model complexity and training time.

| Method | Feature Count | Accuracy | Training Time |
| :--- | :--- | :--- | :--- |
| **Baseline (All Features)** | 561 | 92.67% | ~21.9 sec |
| **Filter (ANOVA)** | 100 | 88.76% | ~9.3 sec |
| **Embedded (Random Forest)** | **97** | **90.49%** | **~7.9 sec** |
| **Wrapper (RFE)** | 100 | 91.92% | ~60.3 sec |

> 🏆 **Conclusion:** The **Embedded Method** provided the best balance between speed and accuracy, making it the most efficient choice for resource-constrained environments like mobile devices.

### 📄 Detailed Report
For a comprehensive analysis including methodology, domain knowledge interpretation, and literature comparison, please read the full report:

👉 **[Read Full Project Report](PROJECT_REPORT.pdf)**

### 🚀 How to Run
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/z-ozturk/Human_Activity_Recognition](https://github.com/z-ozturk/Human_Activity_Recognition)

2. **Install dependencies:**
  ```bash
  pip install pandas numpy scikit-learn matplotlib seaborn

3. **Run the script:**
  ```bash
  python feature_selection_comparison.py
   ---

## 🇹🇷 Türkçe Açıklama

### 📌 Proje Özeti
Bu proje, **UCI İnsan Aktivite Tanıma (HAR)** veri seti kullanılarak farklı **Öznitelik Seçimi yöntemlerinin** (Filtre, Gömülü, Sarmalama) karşılaştırmalı analizini gerçekleştirir.

Temel amaç, 561 boyutlu yüksek öznitelik uzayını daraltırken; Yürüme, Oturma ve Ayakta Durma gibi insan aktiviteleri için yüksek sınıflandırma doğruluğunu korumaktır.

### 🛠 Kullanılan Teknolojiler
* **Dil:** Python 3.x
* **Kütüphaneler:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
* **Teknikler:** * ANOVA F-Değeri (Filtre)
  * Random Forest Önem Derecesi (Gömülü)
  * Recursive Feature Elimination - RFE (Sarmalama)

### 📊 Temel Sonuçlar
Analizler, öznitelik sayısının **561'den ~100'e** düşürülmesinin doğruluk oranında çok az bir kayba neden olurken, model karmaşıklığını ve eğitim süresini önemli ölçüde azalttığını göstermiştir.

| Yöntem | Öznitelik Sayısı | Doğruluk | Eğitim Süresi |
| :--- | :--- | :--- | :--- |
| **Baz Model (Tüm Öznitelikler)** | 561 | %92.67 | ~21.9 sn |
| **Filtre (ANOVA)** | 100 | %88.76 | ~9.3 sn |
| **Gömülü (Random Forest)** | **97** | **%90.49** | **~7.9 sn** |
| **Sarmalama (RFE)** | 100 | %91.92 | ~60.3 sn |

> 🏆 **Sonuç:** **Gömülü Yöntem (Embedded Method)**, hız ve doğruluk arasındaki en iyi dengeyi sağlayarak, mobil cihazlar gibi kaynak kısıtlı ortamlar için en verimli seçenek olduğunu kanıtlamıştır.

### 📄 Detaylı Rapor
Metodoloji, alan bilgisi (domain knowledge) yorumları ve literatür karşılaştırmasını içeren kapsamlı analiz için lütfen tam raporu okuyunuz:

👉 **[Proje Raporunun Tamamını Oku](PROJECT_REPORT.pdf)**

### 🚀 Nasıl Çalıştırılır?
1. **Repoyu klonlayın:**
   ```bash
   git clone [https://github.com/z-ozturk/Human_Activity_Recognition](https://github.com/z-ozturk/Human_Activity_Recognitiont)

2. **Gerekli kütüphaneleri yükleyin:**
  ```bash
  pip install pandas numpy scikit-learn matplotlib seaborn

3. **Analiz kodunu çalıştırın:**
  ```bash
  python feature_selection_comparison.py
   ---