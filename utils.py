#import numpy as np
#import pandas as pd
#import regex as re
#import joblib
#import en_core_web_sm

#from sklearn.base import BaseEstimator, TransformerMixin
#from sklearn.svm import LinearSVC

#nlp = en_core_web_sm.load()
#classifier = LinearSVC()

#def clean_text(text):
    # reduce multiple spaces and newlines to only one
  #  text = re.sub(r'(\s\s+|\n\n+)', r'\1', text)
    # remove double quotes
  #  text = re.sub(r'"', '', text)

  #  return text
	
#def convert_text(text):
 #   sent = nlp(text)
  #  ents = {x.text: x for x in sent.ents}
   # tokens = []
    #for w in sent:
     #   if w.is_stop or w.is_punct:
      #      continue
       # if w.text in ents:
        #    tokens.append(w.text)
      #  else:
       #     tokens.append(w.lemma_.lower())
   # text = ' '.join(tokens)

    #return text


#class preprocessor(TransformerMixin, BaseEstimator):

 #   def __init__(self):
  #      pass

#    def fit(self, X, y=None):
 #       return self

#    def transform(self, X):
 #       return X.apply(clean_text).apply(convert_text)#


import numpy as np
import pandas as pd
import regex as re
import joblib
import en_core_web_sm

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

# Load spaCy model
nlp = en_core_web_sm.load()

# Your classifier
classifier = LinearSVC()

# Text cleaner
def clean_text(text):
    text = re.sub(r'(\s\s+|\n\n+)', r'\1', text)
    text = re.sub(r'"', '', text)
    return text

# Text converter (lemmatize, remove stopwords/punctuation)
def convert_text(text):
    sent = nlp(text)
    ents = {x.text: x for x in sent.ents}
    tokens = []
    for w in sent:
        if w.is_stop or w.is_punct:
            continue
        if w.text in ents:
            tokens.append(w.text)
        else:
            tokens.append(w.lemma_.lower())
    return ' '.join(tokens)

# Custom transformer
class preprocessor(BaseEstimator, TransformerMixin):
    def _init_(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.apply(clean_text).apply(convert_text)


# Main training code
if _name_ == "_main_":
    # Simple inline language dataset
    data = [
        ("Bonjour, comment ça va ?", "French"),
        ("Hello, how are you?", "English"),
        ("Hola, ¿cómo estás?", "Spanish"),
        ("Wie geht es dir?", "German"),
        ("Ciao, come stai?", "Italian"),
        ("Salam, kaifa haluk?", "Arabic"),
        ("Привет, как дела?", "Russian"),
        ("こんにちは、お元気ですか？", "Japanese"),
        ("안녕하세요, 잘 지내세요?", "Korean"),
        ("Merhaba, nasılsın?", "Turkish")
    ]

    df = pd.DataFrame(data, columns=["text", "language"])

    # Build pipeline with your custom preprocessor
    pipe = make_pipeline(preprocessor(), TfidfVectorizer(), classifier)

    # Train the model
    pipe.fit(df['text'], df['language'])

    # Save model
    joblib.dump(pipe, open('language_model.joblib', 'wb'))
