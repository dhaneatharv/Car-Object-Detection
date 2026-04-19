import tensorflow as tf
import numpy as np
import os
from model import build_model

# ✅ Use absolute path to avoid any folder issues
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, 'saved_model')

# ✅ Force create the folder
if os.path.exists(SAVE_DIR):
    print(f"Save folder found: {SAVE_DIR}")
else:
    os.makedirs(SAVE_DIR)
    print(f"Created save folder: {SAVE_DIR}")

# Load preprocessed data
X_train = np.load(os.path.join(BASE_DIR, 'X_train.npy'))
X_val   = np.load(os.path.join(BASE_DIR, 'X_val.npy'))
y_train = np.load(os.path.join(BASE_DIR, 'y_train.npy'))
y_val   = np.load(os.path.join(BASE_DIR, 'y_val.npy'))

optimizers = {
    'Adam':    tf.keras.optimizers.Adam(learning_rate=0.001),
    'SGD':     tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
    'RMSprop': tf.keras.optimizers.RMSprop(learning_rate=0.001),
    'Adagrad': tf.keras.optimizers.Adagrad(learning_rate=0.01)
}

all_histories = {}

for name, opt in optimizers.items():
    print(f"\nTraining with {name} optimizer...")
    model = build_model()
    model.compile(optimizer=opt, loss='mse', metrics=['mae'])

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=32,
        verbose=1
    )

    all_histories[name] = history.history

    # ✅ Full absolute save path
    save_path = os.path.join(SAVE_DIR, f'model_{name}.keras')
    model.save(save_path)
    print(f"✅ {name} saved to {save_path}")
    print(f"   Best val_loss: {min(history.history['val_loss']):.4f}")

np.save(os.path.join(BASE_DIR, 'all_histories.npy'), all_histories)
print("\n✅ All models trained and saved!")