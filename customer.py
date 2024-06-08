from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# Load pre-trained sentiment-analysis model
sentiment_model = pipeline('sentiment-analysis')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    review = data['review']
    result = sentiment_model(review)[0]
    sentiment = result['label']
    confidence = result['score']
    return jsonify(sentiment=sentiment, confidence=confidence)

if __name__ == '__main__':
    app.run(debug=True)
