import streamlit as st
import joblib

def run():
    # Load the full pipeline model (preprocessor + vectorizer + classifier)
    model = joblib.load(open('language_model.joblib', 'rb'))

    st.title("Language Identifier 🌐")
    st.text("Detect the language of a written sentence.")
    st.text("")
    userinput = st.text_input('Enter a sentence:', placeholder='e.g., Bonjour, comment ça va ?')
    st.text("")
    if st.button("Identify") and userinput:
        # Directly pass input to pipeline
        prediction = model.predict([userinput])[0]
        result = f'The detected language for: "{userinput}" is *{prediction}*.'
        st.success(result)

if _name_ == "_main_":
    run()
