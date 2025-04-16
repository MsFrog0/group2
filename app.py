import streamlit as st
import joblib
import pandas as pd

def run():
    model = joblib.load(open('news_classifier_model.joblib', 'rb'))

    st.title("Fake News Detection")
    st.text("Simple app to classify if a news article is real or fake.")
    st.text("")
    
    userinput = st.text_area('Enter a news article snippet below:', placeholder='e.g., The government announced a new policy...')
    st.text("")
    
    prediction_result = ""
    if st.button("Classify"):
        prediction_result = model.predict(pd.Series(userinput))[0]
        if prediction_result == 1:
            output = 'Real 📰'
        else:
            output = 'Fake 🚨'
        classification = f'The entered news text is classified as: {output}'
        st.success(classification)

if __name__ == "__main__":
    run()
