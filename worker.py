import os
import string
import psycopg2
from google.cloud import storage
from flask import Flask, request, jsonify
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

app = Flask(__name__)
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'toekangku-credentials.json'
storage_client = storage.Client()

# Load the pre-trained text classification model
model = tf.keras.models.load_model('model_worker.h5')

# Tokenizer configuration
vocab_size = 10000
oov_token = "<OOV>"
max_length = 100
trunc_type = 'post'

def query_database(classification_result):
    # Establish a connection to the database
    conn = psycopg2.connect(
        dbname="db_toekangku",
        user="postgres",
        password="T03K4NGKU",
        host="34.101.160.78",
        port="5432"
    )

    cursor = conn.cursor()

    # Construct SQL query to match classification result with table column
    query = "SELECT * FROM public.tes_api2 WHERE kategori = %s"
    cursor.execute(query, (classification_result,))

    # Fetch rows/ids matching the classification result
    matched_ids = [row[0] for row in cursor.fetchall()]

    cursor.close()
    conn.close()

    return matched_ids
    
# Helper functions for text preprocessing
def remove_punctuation(sentences):
    translator = str.maketrans('', '', string.punctuation)
    no_punct = sentences.translate(translator)
    return no_punct

def remove_stopword(sentences):
    sentences = sentences.lower()
    sentences = remove_punctuation(sentences)
    factory = StopWordRemoverFactory()
    stopwords = factory.get_stop_words()
    words = sentences.split()
    words_result = [word for word in words if word not in stopwords]
    sentences = ' '.join(words_result)
    return sentences

# API endpoint for text classification
@app.route('/classify', methods=['POST'])
def classify_text():
    if request.method == 'POST':
        try:
            text_bucket = storage_client.get_bucket(
                'ml-toekangku'
            )
            filename = request.json['filename']
            text_blob = text_bucket.blob('text_uploads/' + filename)
            text_content = text_blob.download_as_text()

            preprocessed_text = remove_stopword(text_content)

            # Tokenize and pad the input text
            tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
            tokenizer.fit_on_texts([preprocessed_text])
            sequence = tokenizer.texts_to_sequences([preprocessed_text])
            padded_sequence = pad_sequences(sequence, maxlen=max_length, truncating=trunc_type, padding='post')

            # Make a prediction
            prediction = model.predict(padded_sequence)
            predicted_class = int(tf.argmax(prediction, axis=1)[0].numpy())

            # Convert the prediction to human-readable form
            classes = ['service kulkas', 'service soundsystem', 'tukang batu', 'tukang semen', 'service sistem', 'tukang las', 'tukang cat', 'service perangkat lunak', 'service keamanan', 'service ac', 'tukang kayu', 'service mesin cuci', 'service televisi', 'service data', 'service perangkat keras']
            #classes = ['service ac', 'service kulkas', 'service mesin cuci', 'service soundsystem', 'service televisi', 'service data', 'service keamanan', 'service perangkat keras', 'service perangkat lunak', 'service sistem', 'tukang batu', 'tukang cat', 'tukang kayu', 'tukang las', 'tukang semen']
 
            predicted_category = classes[predicted_class]

            result = query_database(predicted_category)

            return jsonify(result)

        except Exception as e:
                respond = jsonify({'message': f'Error loading text file: {str(e)}'})
                respond.status_code = 400
                return respond
        
    return 'OK'

# Default route for health check
@app.route('/')
def health_check():
    return 'OK'

# Run the application
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(debug=True, host='0.0.0.0', port=port)