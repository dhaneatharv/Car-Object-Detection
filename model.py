# Defines the ResNet50 model — imported by train.py
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout

def build_model():
    base_model = ResNet50(weights='imagenet', 
                          include_top=False, 
                          input_shape=(224, 224, 3))

    for layer in base_model.layers:
        layer.trainable = False  # Freeze ResNet layers

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(4, activation='sigmoid')(x)  # x, y, w, h

    return Model(inputs=base_model.input, outputs=output)

if __name__ == '__main__':
    model = build_model()
    model.summary()  # Just prints model architecture

# HOW TO RUN:
# python model.py   ← optional, just to verify architecture