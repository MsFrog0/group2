import streamlit as st
import joblib
import pandas as pd

def run():
    model = joblib.load(open('language_detector_model.joblib', 'rb'))

    st.title("Language Detector (Arabic or English)")
    st.text("Enter text and this app will tell you if it's in Arabic or English.")
    st.text("")
    
    userinput = st.text_area('Enter your text below:', placeholder='e.g., Hello, how are you?')
    st.text("")
    
    if st.button("Detect Language"):
        if userinput.strip() != "":
            prediction = model.predict(pd.Series([userinput]))[0]
            if prediction == "ar":
                lang = "Arabic 🇸🇦"
            elif prediction == "en":
                lang = "English 🇺🇸"
            else:
                lang = "Unknown ❓"
            st.success(f"The detected language is: {lang}")
        else:
            st.warning("Please enter some text.")

if __name__ == "__main__":
    run()
