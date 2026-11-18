import matplotlib.pyplot as plt
import numpy as np

# 1. Veri Setimiz (Ders Saati -> Sınav Notu)
X = np.array([1, 2, 3, 5, 6, 7, 9])
Y = np.array([35, 40, 50, 65, 75, 80, 95])

# 2. Başlangıç Parametreleri
w = 0.0
b = 0.0

# 3. Hiperparametreler
learning_rate = 0.01
epochs = 1000
m = len(X)

print(f"Başlangıç -> w: {w}, b: {b}")

# Görselleştirme için geçmiş verileri tutacağımız listeler
loss_history = []
w_history = []
b_history = []

# 4. Eğitim (Gradient Descent) Döngüsü
for epoch in range(epochs):
    # İleri Yayılım (Tahmin)
    y_pred = w * X + b

    # Hata ve parametre kaydı
    loss = np.mean((y_pred - Y) ** 2)
    loss_history.append(loss)
    w_history.append(w)
    b_history.append(b)

    # Kısmi Türevler (Gradients)
    dw = (2 / m) * np.sum((y_pred - Y) * X)
    db = (2 / m) * np.sum(y_pred - Y)

    # Parametre Güncelleme
    w = w - learning_rate * dw
    b = b - learning_rate * db

# --- DÖNGÜ BİTTİ ---
print("\nEğitim Tamamlandı!")
print(f"Bulunan w: {w:.4f}")
print(f"Bulunan b: {b:.4f}")

# 5. Görselleştirme
plt.figure(figsize=(12, 5))

# Sol grafik: Hata zamanla nasıl düştü
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs + 1), loss_history, color="red", linewidth=2)
plt.title("Epoch vs MSE Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE (Hata)")
plt.grid(True, linestyle="--", alpha=0.5)

# Sağ grafik: Doğrunun veriye oturması
plt.subplot(1, 2, 2)
plt.scatter(X, Y, color="blue", label="Gerçek Veri", zorder=5)

x_vals = np.array([0, 10])
plt.plot(
    x_vals,
    w_history[0] * x_vals + b_history[0],
    "g--",
    label="Epoch 1 (Başlangıç)",
)
plt.plot(
    x_vals,
    w_history[10] * x_vals + b_history[10],
    "m--",
    label="Epoch 11 (İlerleme)",
)
plt.plot(
    x_vals,
    w * x_vals + b,
    "r-",
    label=f"Epoch {epochs} (Son Hal)",
    linewidth=2,
)

plt.title("Doğrunun Veriye Oturma Süreci")
plt.xlabel("Çalışma Saati")
plt.ylabel("Sınav Notu")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("loss_curve.png", dpi=300)  # GitHub README için görseli kaydeder
plt.show()