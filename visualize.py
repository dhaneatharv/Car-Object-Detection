# Plots optimizer comparison graphs + draws bounding boxes
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf, cv2

# ── Plot 1: Optimizer Comparison ──
histories = np.load('all_histories.npy', allow_pickle=True).item()

plt.figure(figsize=(12, 5))
for name, h in histories.items():
    plt.plot(h['val_loss'], label=name)
plt.title('Optimizer Comparison - Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.legend()
plt.savefig('optimizer_comparison.png')
plt.show()

# ── Plot 2: Draw bounding box on a test image ──
model = tf.keras.models.load_model('saved_model/model_Adam.keras')  # ✅

img = cv2.imread('data/images/test_car.jpg')
img_resized = cv2.resize(img, (224, 224)) / 255.0
pred = model.predict(np.expand_dims(img_resized, axis=0))[0]

h, w = img.shape[:2]
x, y, bw, bh = (pred * [w, h, w, h]).astype(int)
cv2.rectangle(img, (x, y), (x+bw, y+bh), (0, 255, 0), 2)
cv2.imwrite('detection_result.jpg', img)
print("Detection result saved!")

# HOW TO RUN:
# python visualize.py