import streamlit as st
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
import tensorflow as tf
import google.generativeai as palm
from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv("API_KEY")

# Tokenization and Model Prediction
file_path = r"C:\Users\HP\Desktop\News Navigator\streamlit\saved_models"
tokenizer = DistilBertTokenizer.from_pretrained(file_path)
model = TFDistilBertForSequenceClassification.from_pretrained(file_path)

# Streamlit app
st.title("News Navigator 🧭")
user_input = st.text_area("Enter News Articles (separated by double line breaks):")
classify_button = st.button("Classify")
result_placeholder = st.empty()
main_placeholder = st.empty()

# Define categories before using them
categories = ["Business", "Entertainment", "Politics", "Sports", "Tech"]

# Move this block inside the if block
if classify_button and user_input:
    # Split the input into a list of articles
    articles = [article.strip() for article in user_input.split('$')]

    # Process each article
    for idx, article in enumerate(articles, start=1):  # Start index at 1
        # Tokenize and predict category
        inputs = tokenizer(article,
                           truncation=True,
                           padding=True,
                           return_tensors="tf")
        outputs = model(inputs["input_ids"])[0]
        predicted_category = tf.argmax(outputs, axis=1).numpy()[0]

        # Ensure predicted_category is within the valid range of categories
        predicted_category = min(predicted_category, len(categories) - 1)

        # Map numerical category to label (adjust as needed)
        predicted_label = categories[predicted_category]

        # Display the result with HTML styling using st.markdown
        st.markdown(
            f"<h2 style='color:#4A712F  ;'>The predicted category for Article {idx} is: {predicted_label}</h2>",
            unsafe_allow_html=True)

        # Use the article in the prompt for summary generation
        palm.configure(api_key=API_KEY)
        model_id = 'models/text-bison-001'

        prompt = f'''I will give you a news article. Study the news article and give a summary of it within 100 words.\n{article}'''

        completion = palm.generate_text(
            model=model_id,
            prompt=f"{article}\n{prompt}",
            temperature=0.0,
            max_output_tokens=500,
            candidate_count=1)

        summary = completion.result

        # Display the summary on the Streamlit app
        st.markdown(
            f"<p style='font-size:18px; color:#fc3535;'>Generated Summary for Article {idx}:</p><p style='font-size:16px;'>{summary}</p>",
            unsafe_allow_html=True)
