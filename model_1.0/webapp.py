import streamlit as st
import requests
import json

st.set_page_config(page_title="Hinglish Sentiment AI", page_icon="🔥", layout="centered")

st.title("🔥 Hinglish Sentiment Engine")
st.markdown("Enter code-mixed Hindi-English (Hinglish) with typos, shorthand, and emojis.")

API_URL = "http://127.0.0.1:8000/predict"

user_input = st.text_area("Type your message here:", placeholder="Bhai kya solid update diya hai team ne, dil jeet liya ❤️")

if st.button("Analyze Sentiment"):
    if user_input.strip():
        try:
            with st.spinner("Running ONNX Inference..."):
                response = requests.post(API_URL, json={"text": user_input})
                response.raise_for_status()
                data = response.json()

            sentiment = data["sentiment"]
            conf = data["confidence"] * 100

            # Dynamic coloring
            if sentiment == "Positive":
                color = "green"
                emoji = "✅"
            elif sentiment == "Negative":
                color = "red"
                emoji = "🚩"
            else:
                color = "gray"
                emoji = "⚖️"

            st.markdown(f"### Result: <span style='color:{color}'>{emoji} {sentiment}</span>", unsafe_allow_html=True)
            st.progress(data["confidence"])
            st.write(f"**Confidence:** {conf:.2f}%")

            st.info(f"⚡ **Latency:** {data['latency_ms']} ms | 💻 **Hardware:** {data['device_used'].upper()}")

        except Exception as e:
            st.error(f"Failed to connect to backend API. Is FastAPI running? Error: {e}")
    else:
        st.warning("Please enter some text to analyze.")
