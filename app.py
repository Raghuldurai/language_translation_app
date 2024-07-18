from flask import Flask, request, render_template
from transformers import MarianMTModel, MarianTokenizer
from dotenv import load_dotenv
import os

app = Flask(__name__)

# Load environment variables
load_dotenv()

# Get the model_name from environment variable
model_name = os.getenv('model_name')

# Initialize tokenizer and model
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# Function for translation
def translate(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    outputs = model.generate(**inputs)
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated_text

@app.route('/', methods=['GET', 'POST'])
def index():
    translated_text = ''
    if request.method == 'POST':
        text_to_translate = request.form['text']
        translated_text = translate(text_to_translate, model, tokenizer)
    return render_template('index.html', translated_text=translated_text)

if __name__ == "__main__":
    app.run(debug=True)
