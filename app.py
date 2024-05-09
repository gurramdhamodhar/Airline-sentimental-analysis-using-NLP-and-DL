import pandas as pd
from flask import Flask, render_template, request
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# Load DataFrame
df = pd.read_csv(r"C:\Users\gurra\Desktop\Airline\Tweets.csv")

# Filter out neutral sentiment tweets
tweet_df = df[df['airline_sentiment'] != 'neutral'][['text', 'airline_sentiment']]

# Factorize sentiment labels
sentiment_label = tweet_df['airline_sentiment'].factorize()

# Tokenization and padding
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(tweet_df['text'])
vocab_size = len(tokenizer.word_index) + 1

# Load the model
model = load_model(r"C:\Users\gurra\Desktop\Airline\dhammu.h5")

# Function to predict sentiment
def predict_sentiment(text):
    try:
        tw = tokenizer.texts_to_sequences([text])
        tw = pad_sequences(tw, maxlen=200)
        prediction = int(model.predict(tw).round().item())
        return sentiment_label[1][prediction]
    except Exception as e:
        print("Error predicting sentiment:", str(e))
        return "Error"

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/Predict', methods=['POST', 'GET'])
def Predict():
    if request.method == "POST":
        test_sentence = request.form['Review']
        prediction = predict_sentiment(test_sentence)
        return render_template("output.html", result=prediction)
    return render_template("output.html", result="")

if __name__ == "__main__":
    app.run(debug=True)
