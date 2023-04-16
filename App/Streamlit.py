import streamlit as st
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# Load model
model = tf.keras.models.load_model('App/model.h5')

# Load data
notes_df = pd.DataFrame({'note': ['']*30, 'sentiment': [0.6]*30})

with open('App/tokenizer.json', 'r', encoding='utf-8') as f:
    tokenizer_json = f.read()
tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_json)

max_length = 100

# Define function to predict sentiment of text
def predict_sentiment(text):
    text = [text]

    text_sequence = tokenizer.texts_to_sequences(text)
    text_padded = pad_sequences(text_sequence, padding='post', maxlen=100)
    return model.predict(text_padded)[0]



def add_and_view_notes():
    global notes_df
    with st.container():
        st.write("## Notes:")
        consecutive_days = 0
        total_days = 0
        alert_displayed = False
        for i in range(0, 30):
            with st.expander(f"April {i+1}"):
                note = st.text_area("Add a note:", key=f"note_{i}")
                if st.button("Save note", key=f"save_{i}"):
                    notes_df.loc[i, 'note'] = note
                    notes_df.loc[i, 'sentiment'] = predict_sentiment(note)
                    sentiment = notes_df.loc[i, 'sentiment']
                    color = 'red' if sentiment >= 0.6 else 'green'
                    st.markdown(f"<p style='background-color: {color}'>{note}</p>", unsafe_allow_html=True)
                    if color == 'red' and not alert_displayed:
                        if st.button("Would you like to connect with our therapist?", key=f"alert_{i}"):
                            st.write("Connecting with therapist...")
                            st.markdown(f'<a href="{'https://cerina.co/#footer'}">{button_label}</a>', unsafe_allow_html=True)
                
                            alert_displayed = True
                    else:
                        alert_displayed = False
                else:
                    sentiment = predict_sentiment(note)
                    notes_df.loc[i, 'sentiment'] = sentiment
                    color = 'red' if sentiment >= 0.6 else 'green'
                    st.markdown(f"<p style='background-color: {color}'>{note}</p>", unsafe_allow_html=True)




# Set page title
st.set_page_config(page_title="Mental Health Platform")

# Define app layout
def app():
    st.title("Cerina - Mental Health Care")
    add_and_view_notes()

if __name__ == "__main__":
    app()
