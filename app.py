import streamlit as st
import pandas as pd
import joblib
from utils import preprocessor

def run():
    st.title("Language Identification")
    st.text("This app detects the language of the text you enter.\n")

    # Load model
    model = joblib.load('language_identifier_model.joblib')

    # Load and show supported languages
    label_encoder = joblib.load('label_encoder.joblib')
    supported_languages = label_encoder.classes_
    st.markdown("**Supported languages:**")
    st.write(", ".join(supported_languages))
    st.text("")

    # User input
    userinput = st.text_input(
        'Type or paste some text:',
        placeholder='e.g., Bonjour, comment ça va?'
    )
    st.text("")

    # Button click
    if st.button("Detect Language"):
        if userinput.strip() != "":
            cleaned_text = preprocessor(userinput)
            predicted_lang = model.predict(pd.Series([cleaned_text]))[0]
            st.success(f'The detected language is: **{predicted_lang}**')
        else:
            st.warning("Please enter some text to detect.")

if __name__ == "__main__":
    run()
