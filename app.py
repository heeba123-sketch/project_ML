import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf

# ------------------------------------------------------------------
# Config générale
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Détecteur Fruits & Légumes 🍎🥕",
    page_icon="🍏",
    layout="centered",
)

MODEL_PATH = "best_model_finetuned.tflite"
IMG_SIZE = (224, 224)

# Ordre EXACT des classes tel qu'utilisé pendant l'entraînement
# (tri alphabétique de os.listdir sur le dataset équilibré)
CLASS_NAMES = [
    "Apple__Healthy", "Apple__Rotten",
    "Banana__Healthy", "Banana__Rotten",
    "Bellpepper__Healthy", "Bellpepper__Rotten",
    "Carrot__Healthy", "Carrot__Rotten",
    "Cucumber__Healthy", "Cucumber__Rotten",
    "Grape__Healthy", "Grape__Rotten",
    "Guava__Healthy", "Guava__Rotten",
    "Jujube__Healthy", "Jujube__Rotten",
    "Mango__Healthy", "Mango__Rotten",
    "Orange__Healthy", "Orange__Rotten",
    "Pomegranate__Healthy", "Pomegranate__Rotten",
    "Potato__Healthy", "Potato__Rotten",
    "Strawberry__Healthy", "Strawberry__Rotten",
    "Tomato__Healthy", "Tomato__Rotten",
]

FR_NAMES = {
    "Apple": "Pomme", "Banana": "Banane", "Bellpepper": "Poivron",
    "Carrot": "Carotte", "Cucumber": "Concombre", "Grape": "Raisin",
    "Guava": "Goyave", "Jujube": "Jujube", "Mango": "Mangue",
    "Orange": "Orange", "Pomegranate": "Grenade", "Potato": "Pomme de terre",
    "Strawberry": "Fraise", "Tomato": "Tomate",
}


def pretty_label(raw_label: str) -> str:
    fruit, state = raw_label.split("__")
    fruit_fr = FR_NAMES.get(fruit, fruit)
    state_fr = "Sain" if state == "Healthy" else "Pourri / Abîmé"
    return f"{fruit_fr} — {state_fr}"


@st.cache_resource
def load_interpreter():
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter


def preprocess(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(image).astype("float32")
    # Le modèle est basé sur EfficientNetB0 : la normalisation (rescaling)
    # est intégrée dans le graphe du modèle, donc on n'applique PAS de
    # division par 255 ici, on garde les valeurs brutes 0-255.
    arr = np.expand_dims(arr, axis=0)
    return arr


def predict(interpreter, image_array: np.ndarray):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]["index"], image_array)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])[0]
    return output


# ------------------------------------------------------------------
# Interface
# ------------------------------------------------------------------
st.title("🍎🥕 Détecteur de fruits & légumes sains ou pourris")
st.write(
    "Envoie une photo d'un fruit ou d'un légume, le modèle (EfficientNetB0 "
    "fine-tuné, format TFLite) prédit s'il est **sain** ou **pourri**, "
    "parmi 14 types de fruits/légumes (28 classes)."
)

with st.spinner("Chargement du modèle..."):
    interpreter = load_interpreter()

uploaded_file = st.file_uploader(
    "Choisis une image (jpg, jpeg, png)", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Image envoyée", use_container_width=True)

    with st.spinner("Analyse en cours..."):
        arr = preprocess(image)
        probs = predict(interpreter, arr)

    top_idx = np.argsort(probs)[::-1][:5]

    with col2:
        best_idx = top_idx[0]
        st.subheader(pretty_label(CLASS_NAMES[best_idx]))
        st.metric("Confiance", f"{probs[best_idx] * 100:.1f} %")

    st.markdown("### Top 5 prédictions")
    for idx in top_idx:
        st.write(f"**{pretty_label(CLASS_NAMES[idx])}** — {probs[idx] * 100:.2f} %")
        st.progress(float(probs[idx]))
else:
    st.info("👆 Envoie une image pour lancer la prédiction.")

st.divider()
st.caption(
    "Modèle entraîné sur le dataset Kaggle "
    "*Fruit And Vegetable Diseases Dataset* (28 classes, EfficientNetB0)."
)
