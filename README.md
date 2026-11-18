# Matrix-Based Regression & Optimization Engine from Scratch

Bu depo, **Yapay Sinir Ağları (Artificial Neural Networks)** dersi kapsamında hazır makine öğrenmesi kütüphaneleri (`scikit-learn`, `PyTorch`, `TensorFlow`) kullanılmadan, tamamen saf **Python** ve **NumPy** matris operasyonları ile sıfırdan geliştirilen optimizasyon ve regresyon algoritmalarını içerir.

## 🎯 Projenin Amacı ve Kapsamı
Derin öğrenme ve yapay sinir ağlarının temelini oluşturan ileri yayılım (forward pass), kayıp fonksiyonu (MSE), kısmi türev hesaplamaları (gradients) ve parametre güncelleme adımlarının matematiksel altyapısını düşük seviyede (low-level) uygulamak.

## ⚙️ Uygulanan Yöntemler
* **Vektörize Matris İşlemleri (`NumPy`):** Döngü maliyeti olmadan $X_{scaled} \cdot W + b$ matris çarpımı (`@`) ve transpoze (`.T`) operasyonlarıyla yüksek performanslı hesaplama.
* **Veri Ön İşleme (Z-Score Feature Scaling):** Farklı ölçeklerdeki giriş özelliklerinin ($\mu = 0, \sigma = 1$) standardizasyonu.
* **Overfitting Kontrolü (L2 Regularization / Ridge):** Ağırlıkların aşırı büyümesini engellemek amacıyla kayıp fonksiyonuna ve gradyan hesaplamasına $\frac{\lambda}{m} W$ ceza teriminin entegrasyonu.
* **Manuel Gradient Descent:** Öğrenme oranı ($\alpha$) ve epoch sayısına bağlı ağırlık ($W$) ve bias ($b$) yakınsama optimizasyonu.
* **Performans Görselleştirme (`Matplotlib`):** Eğitim süreci boyunca Ortalama Kare Hata (MSE) düşüş eğrilerinin analizi.

## 📂 Dosya Yapısı
* `01_linear_regression_gd.py`: Tek değişkenli doğrusal regresyon, manuel türev hesabı ve kayıp (loss) eğrisi görselleştirmesi.
* `02_multiple_regression_l2.py`: Çok değişkenli özellik matrisi, Z-Score normalizasyonu, L2 (Ridge) cezalı gradyan inişi ve yeni veri üzerinde çıkarım (inference).

## 🚀 Çalıştırma
```bash
pip install numpy matplotlib
python 02_multiple_regression_l2.py
