# Loads saved models and prints accuracy comparison table
import numpy as np
import tensorflow as tf

X_val  = np.load('X_val.npy')
y_val  = np.load('y_val.npy')

optimizer_names = ['Adam', 'SGD', 'RMSprop', 'Adagrad']

print(f"\n{'Optimizer':<12} {'Val Loss (MSE)':<18} {'Val MAE'}")
print("-" * 42)

for name in optimizer_names:
    model = tf.keras.models.load_model(f'saved_model/model_{name}.keras')  # ✅
    loss, mae = model.evaluate(X_val, y_val, verbose=0)
    print(f"{name:<12} {loss:<18.4f} {mae:.4f}")

# HOW TO RUN:
# python evaluate.py