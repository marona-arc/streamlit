import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"
EMOJI_MAP = {
    "anger": "😠",
    "disgust": "🤢",
    "fear": "😨",
    "joy": "😊",
    "neutral": "😐",
    "sadness": "😢",
    "surprise": "😮",
}


@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    return tokenizer, model


def predict_emotion(text: str):
    tokenizer, model = load_model()
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(outputs.logits, dim=-1)[0]
    labels = model.config.id2label
    results = {
        labels[i]: float(probabilities[i].item())
        for i in range(len(labels))
    }
    predicted_label = max(results, key=results.get)
    return results, predicted_label


st.set_page_config(page_title="Emotion Detector", page_icon="😊", layout="centered")

st.title("Emotion Detector")
st.caption("Analyze the emotion of a sentence using a Hugging Face transformer model.")

sample_texts = [
    "I'm really excited about this project and the progress we made.",
    "I feel so anxious and worried about the upcoming deadline.",
    "This is such a surprise and I can't believe it happened.",
    "I'm feeling a little neutral about this situation.",
]

with st.sidebar:
    st.header("Examples")
    for example in sample_texts:
        if st.button(example[:60] + ("..." if len(example) > 60 else ""), use_container_width=True):
            st.session_state.text_input = example

user_input = st.text_area(
    "Enter text to analyze",
    key="text_input",
    value=st.session_state.get("text_input", ""),
    height=180,
    placeholder="Type a sentence here...",
)

col1, col2 = st.columns([1, 1])
with col1:
    analyze = st.button("Analyze emotion", use_container_width=True)
with col2:
    if st.button("Clear", use_container_width=True):
        st.session_state.text_input = ""

if analyze:
    if not user_input.strip():
        st.warning("Please enter some text before analyzing.")
    else:
        with st.spinner("Running emotion analysis..."):
            results, predicted_label = predict_emotion(user_input)

        confidence = results[predicted_label]
        emoji = EMOJI_MAP.get(predicted_label, "💬")

        st.subheader(f"Predicted emotion: {predicted_label.title()} {emoji}")
        st.metric("Confidence", f"{confidence * 100:.2f}%")

        ranked_results = sorted(results.items(), key=lambda item: item[1], reverse=True)
        for label, probability in ranked_results:
            label_emoji = EMOJI_MAP.get(label, "💬")
            st.markdown(f"**{label_emoji} {label.title()}**")
            st.progress(probability)
            st.caption(f"{probability * 100:.2f}%")

        st.markdown("---")
        with st.expander("View all probabilities"):
            st.json(results)
