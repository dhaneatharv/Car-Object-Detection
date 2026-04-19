import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import os

# ─── Page Config ───
st.set_page_config(
    page_title="Car Object Detection",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Car Object Detection using ResNet50")
st.markdown("Upload a car image to detect it, and explore optimizer performance.")

# ─── Sidebar ───
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["🔍 Detect Car", "📊 Optimizer Comparison", "📈 Accuracy Stats"])

# ─── Helper: Load Model ───
@st.cache_resource
def load_model(optimizer_name):
    path = os.path.join('saved_model', f'model_{optimizer_name}.keras')
    if os.path.exists(path):
        return tf.keras.models.load_model(path)
    return None

# ─── Helper: Draw Bounding Box ───
def draw_bbox(image, pred_box):
    img = np.array(image.convert('RGB'))
    h, w = img.shape[:2]
    x, y, bw, bh = pred_box
    x1 = int(x * w)
    y1 = int(y * h)
    x2 = int((x + bw) * w)
    y2 = int((y + bh) * h)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
    cv2.putText(img, 'Car', (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return img

# ════════════════════════════════════════
# PAGE 1: Detect Car
# ════════════════════════════════════════
if page == "🔍 Detect Car":
    st.header("🔍 Car Detection")
    st.markdown("Upload a car image and select an optimizer model to run detection.")

    col1, col2 = st.columns(2)

    with col1:
        uploaded_file = st.file_uploader("Upload Car Image", type=['jpg', 'jpeg', 'png'])
        optimizer_choice = st.selectbox("Select Optimizer Model", ['Adam', 'SGD', 'RMSprop', 'Adagrad'])

        detect_btn = st.button("🚗 Detect Car", use_container_width=True)

    if uploaded_file and detect_btn:
        image = Image.open(uploaded_file)

        with col1:
            st.image(image, caption="Original Image", use_container_width=True)

        with st.spinner(f"Running detection with {optimizer_choice} model..."):
            model = load_model(optimizer_choice)

            if model is None:
                st.error(f"❌ Model for {optimizer_choice} not found! Please run train.py first.")
            else:
                # Preprocess
                img_resized = image.resize((224, 224))
                img_array = np.array(img_resized) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                # Predict
                pred = model.predict(img_array)[0]

                # Draw bbox
                result_img = draw_bbox(image, pred)

                with col2:
                    st.image(result_img, caption=f"Detection Result ({optimizer_choice})", use_container_width=True)
                    st.success("✅ Detection Complete!")
                    st.markdown("### Bounding Box Coordinates")
                    st.json({
                        "x": round(float(pred[0]), 4),
                        "y": round(float(pred[1]), 4),
                        "width": round(float(pred[2]), 4),
                        "height": round(float(pred[3]), 4)
                    })

# ════════════════════════════════════════
# PAGE 2: Optimizer Comparison Chart
# ════════════════════════════════════════
elif page == "📊 Optimizer Comparison":
    st.header("📊 Optimizer Comparison")

    if not os.path.exists('all_histories.npy'):
        st.error("❌ all_histories.npy not found! Please run train.py first.")
    else:
        histories = np.load('all_histories.npy', allow_pickle=True).item()

        tab1, tab2 = st.tabs(["Validation Loss", "Validation MAE"])

        with tab1:
            fig, ax = plt.subplots(figsize=(10, 5))
            for name, h in histories.items():
                ax.plot(h['val_loss'], label=name, linewidth=2)
            ax.set_title('Optimizer Comparison — Validation Loss')
            ax.set_xlabel('Epochs')
            ax.set_ylabel('MSE Loss')
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

        with tab2:
            fig, ax = plt.subplots(figsize=(10, 5))
            for name, h in histories.items():
                ax.plot(h['val_mae'], label=name, linewidth=2)
            ax.set_title('Optimizer Comparison — Validation MAE')
            ax.set_xlabel('Epochs')
            ax.set_ylabel('MAE')
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

# ════════════════════════════════════════
# PAGE 3: Accuracy Stats
# ════════════════════════════════════════
elif page == "📈 Accuracy Stats":
    st.header("📈 Accuracy Stats")

    if not os.path.exists('all_histories.npy'):
        st.error("❌ all_histories.npy not found! Please run train.py first.")
    else:
        histories = np.load('all_histories.npy', allow_pickle=True).item()

        st.markdown("### Best Results per Optimizer")

        # Summary table
        data = []
        for name, h in histories.items():
            best_loss = min(h['val_loss'])
            best_mae  = min(h['val_mae'])
            best_epoch = h['val_loss'].index(best_loss) + 1
            data.append({
                "Optimizer": name,
                "Best Val Loss (MSE)": round(best_loss, 4),
                "Best Val MAE": round(best_mae, 4),
                "Best Epoch": best_epoch
            })

        import pandas as pd
        df = pd.DataFrame(data)

        # Highlight best optimizer
        best_optimizer = df.loc[df['Best Val Loss (MSE)'].idxmin(), 'Optimizer']
        st.dataframe(df, use_container_width=True)
        st.success(f"🏆 Best Optimizer: **{best_optimizer}** with lowest validation loss!")

        # Bar chart
        st.markdown("### Loss Comparison Bar Chart")
        fig, ax = plt.subplots(figsize=(8, 4))
        colors = ['green' if o == best_optimizer else 'steelblue' for o in df['Optimizer']]
        ax.bar(df['Optimizer'], df['Best Val Loss (MSE)'], color=colors)
        ax.set_title('Best Validation Loss per Optimizer')
        ax.set_ylabel('MSE Loss')
        ax.set_xlabel('Optimizer')
        for i, v in enumerate(df['Best Val Loss (MSE)']):
            ax.text(i, v + 0.001, str(v), ha='center', fontweight='bold')
        st.pyplot(fig)