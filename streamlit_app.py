import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

def translate_text(article, source_language, target_language):
    # Importing the pre-trained model
    tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")

    # Mapping language to language code
    lang_code = {
        "hindi": "hin_Deva",
        "french": "fra_Latn",
        "english":"eng_Latn"
        # Add more language codes as needed
    }

    # Appending source language to the input text
    article_with_lang = f"{article} [lang:{lang_code[source_language]}]"

    # Translating the sentence
    inputs = tokenizer(article_with_lang, return_tensors="pt")
    translated_tokens = model.generate(
        **inputs, forced_bos_token_id=tokenizer.lang_code_to_id[lang_code[target_language]], max_length=30
    )

    # Decoding and returning the translated text
    return tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

# Streamlit UI
st.title("Language Translator")

# Input section
st.sidebar.header("Input Options")
input_text = st.text_area("Enter Text to Translate:")
source_language = st.sidebar.selectbox("Select Source Language:", ["english", "hindi", "french"])

# Output section
st.sidebar.header("Output Options")
target_language = st.sidebar.selectbox("Select Target Language:", ["english","hindi", "french"])

# Translate button
if st.button("Translate"):
    if input_text:
        translated_text = translate_text(input_text, source_language, target_language)
        st.success(f"Translated ({target_language}): {translated_text}")
    else:
        st.warning("Please enter text to translate.")