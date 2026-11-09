
import streamlit as st
import joblib
import os

st.set_page_config(
    page_title="Amazon Review Sentiment Analyzer",
    page_icon="🛒",
    layout="centered"
)

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_models():
    vectorizer = joblib.load(
        os.path.join(MODEL_DIR, "deployment_vectorizer.pkl")
    )
    classifier = joblib.load(
        os.path.join(MODEL_DIR, "deployment_classifier.pkl")
    )
    return vectorizer, classifier

vectorizer, classifier = load_models()

st.title("🛒 Amazon Review Sentiment Analyzer")

st.write(
    "Analyze customer reviews using a Bigram TF-IDF "
    "and Logistic Regression machine learning model."
)

st.markdown("---")

review = st.text_area(
    "Enter your product review:",
    placeholder="Example: This product is amazing. Great quality and taste!",
    height=150
)

if st.button("Analyze Sentiment", type="primary", use_container_width=True):

    if not review.strip():
        st.warning("Please enter a review first.")

    else:
        features = vectorizer.transform([review])
        prediction = classifier.predict(features)[0]
        probabilities = classifier.predict_proba(features)[0]

        negative_prob = float(probabilities[0])
        positive_prob = float(probabilities[1])

        st.markdown("### Prediction")

        if prediction == 1:
            st.success("😊 Positive Review")
        else:
            st.error("😞 Negative Review")

        st.markdown("### Prediction Confidence")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Positive",
                f"{positive_prob * 100:.2f}%"
            )

        with col2:
            st.metric(
                "Negative",
                f"{negative_prob * 100:.2f}%"
            )

        st.markdown("### Probability Distribution")

        st.bar_chart({
            "Probability": {
                "Positive": positive_prob,
                "Negative": negative_prob
            }
        })

st.markdown("---")

st.caption(
    "Model: Bigram TF-IDF + Logistic Regression | "
    "Amazon Fine Food Reviews Dataset"
)
