import numpy as np

# 1. Veri Seti: [Çalışma Saati, Uyku Süresi] -> Sınav Notu
X = np.array([
    [1, 8],
    [2, 7],
    [3, 6],
    [5, 8],
    [6, 5],
    [7, 9],
    [9, 4],
])
y = np.array([35, 40, 50, 65, 75, 80, 95])

# 2. Feature Scaling (Z-Score Standardizasyonu)
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X_scaled = (X - X_mean) / X_std

# 3. Başlangıç Parametreleri ve Hiperparametreler
W = np.zeros(2)
b = 0.0
learning_rate = 0.01
lambda_reg = 0.1
epochs = 1000
m = len(X)

# 4. L2 Cezalı (Ridge) Gradient Descent Döngüsü
for epoch in range(epochs):
    # İleri Yayılım (Tahmin)
    y_pred = X_scaled @ W + b

    # Kısmi Türevler (Gradients)
    dW = (1 / m) * (X_scaled.T @ (y_pred - y)) + (lambda_reg / m) * W
    db = (1 / m) * np.sum(y_pred - y)

    # Parametre Güncelleme
    W = W - learning_rate * dW
    b = b - learning_rate * db

# 5. Eğitim Sonuçları
print("--- EĞİTİM TAMAMLANDI ---")
print(f"W1 (Çalışma Saati Ağırlığı) : {W[0]:.4f}")
print(f"W2 (Uyku Süresi Ağırlığı)  : {W[1]:.4f}")
print(f"b (Bias - Taban Not)       : {b:.4f}\n")

# 6. Yeni Veri Üzerinde Çıkarım (Inference)
yeni_ogrenci = np.array([4, 7])
yeni_scaled = (yeni_ogrenci - X_mean) / X_std
tahmin = yeni_scaled @ W + b

print(f"4 Saat Çalışan ve 7 Saat Uyuyan Öğrencinin Tahmini Notu: {tahmin:.2f}")