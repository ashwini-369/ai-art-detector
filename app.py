import cv2
import tensorflow as tf
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ================================
# Page Configuration
# ================================
st.set_page_config(
    page_title="AI vs Human Art Detector",
    layout="wide",
)
st.markdown("""
<style>

/* ===== BACKGROUND ===== */
.stApp {
    background: radial-gradient(circle at 20% 20%, #020617, #000000 70%);
    color: #00ffe1;
    font-family: Consolas, monospace;
}

/* remove streamlit top padding */
.block-container {
    padding-top: 2rem;
    max-width: 95%;
}

/* ===== HEADINGS ===== */
h1 {
    text-align:center;
    color:#00ffe1;
    letter-spacing:2px;
    text-shadow:0 0 8px #00ffe1;
}

h2, h3 {
    color:#22c55e;
    text-shadow:0 0 6px #22c55e;
}

/* ===== FILE UPLOADER ===== */
[data-testid="stFileUploader"] {
    border:1px solid #00ffe1;
    border-radius:10px;
    padding:15px;
    background:#020617;
}

/* ===== BUTTON ===== */
.stButton>button {
    background:black;
    color:#00ffe1;
    border:1px solid #00ffe1;
    border-radius:8px;
    box-shadow:0 0 10px #00ffe1;
    transition:0.3s;
}
.stButton>button:hover {
    background:#00ffe1;
    color:black;
}

/* ===== METRIC BOX ===== */
[data-testid="stMetric"] {
    background:#020617;
    border:1px solid #22c55e;
    padding:15px;
    border-radius:10px;
    box-shadow:0 0 10px #22c55e;
}

/* ===== PROGRESS BAR ===== */
.stProgress > div > div > div {
    background:linear-gradient(90deg,#00ffe1,#22c55e);
}

/* ===== SIDEBAR ===== */
section[data-testid="stSidebar"] {
    background:#000000;
    border-right:1px solid #00ffe1;
}

/* ===== SCROLLBAR ===== */
::-webkit-scrollbar {
    width:8px;
}
::-webkit-scrollbar-thumb {
    background:#00ffe1;
    border-radius:10px;
}

</style>
""", unsafe_allow_html=True)



st.markdown("""
<h1>AI ART AUTHENTICITY SCANNER</h1>
<p style='text-align:center;color:#22c55e;'>
 AI vs Human Art detection
</p>
<hr>
""", unsafe_allow_html=True)



# ================================
# Load Model
# ================================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/resnet50v2_finetuned_full.keras")

model = load_model()

# store history
if "history" not in st.session_state:
    st.session_state.history = []

def make_gradcam_heatmap(img_array, model):
    # find last conv layer safely
    last_conv_layer = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer.name
            break

    if last_conv_layer is None:
        return None

    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(last_conv_layer).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)

    if grads is None:
        return None

    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap)

    if max_val == 0:
        return None

    heatmap /= max_val
    return heatmap.numpy()


# ================================
# Upload image
# ================================
uploaded_file = st.file_uploader("Upload artwork", type=["jpg","jpeg","png"])

if uploaded_file is not None:

    # show image
    image_data = Image.open(uploaded_file).convert("RGB")

    st.image(image_data, caption="Uploaded Image", use_container_width=True)

    # image details
    st.subheader("Image details")
    st.write(f"Filename: {uploaded_file.name}")
    st.write(f"Resolution: {image_data.size[0]} x {image_data.size[1]}")
    st.write(f"Color mode: {image_data.mode}")

    # preprocess
    img = image_data.resize((224,224))
    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner("Running neural scan..."):
        prediction = model.predict(img_array)[0][0]

    # ================= Prediction result =================
    st.divider()
    st.subheader("Prediction Result")

    if prediction > 0.5:
        label = "Human-created artwork"
        confidence = prediction * 100
    else:
        label = "AI-generated artwork"
        confidence = (1 - prediction) * 100

    st.markdown("### Analysis")
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Prediction", label)

    with col2:
        st.metric("Confidence", f"{confidence:.2f}%")

    st.progress(int(confidence))


    # probabilities
    st.subheader("Probabilities")
    st.write(f"AI probability: {(100-confidence):.2f}%")
    st.write(f"Human probability: {confidence:.2f}%")

    # save history
    st.session_state.history.append((uploaded_file.name, label, confidence))
    

    # download result
    result_text = f"Result: {label}\nConfidence: {confidence:.2f}%"
    st.download_button("Download result", result_text, file_name="prediction.txt")
    st.subheader("Model Attention Heatmap")
    heatmap = make_gradcam_heatmap(img_array, model)
    if heatmap is not None:
        heatmap = cv2.resize(heatmap, (image_data.size[0], image_data.size[1]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        original = np.array(image_data)
        superimposed = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)
        st.subheader("Model attention visualization")
        st.image(superimposed, use_container_width=True)

    else:
        st.info("Heatmap unavailable for this image")
    st.subheader("Possible Source Analysis")

    if label == "AI-generated artwork":
        if confidence > 85:
            st.write("Likely generated using advanced diffusion models (Midjourney/DALL·E).")
        else:
            st.write("Possibly generated using basic or edited AI tools.")
    else:
        st.write("No strong AI-generation patterns detected.")





# ================= Footer =================
st.caption("Mini Project by Ashwini | AI Art Detection using ResNet50V2")
