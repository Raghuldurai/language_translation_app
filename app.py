from flask import Flask, request, render_template
from transformers import MarianMTModel, MarianTokenizer

app = Flask(__name__)

# Specify the model and tokenizer for translation (English to French)
model_name = 'Helsinki-NLP/opus-mt-en-fr'
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
