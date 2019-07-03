# Dependencies
import traceback
import model
from flask import Flask, render_template, request, redirect, url_for, flash, get_flashed_messages, jsonify
#from flask_script import Manager, Command, Shell
from forms import SummarizationForm
#from werkzeug.datastructures import MultiDict

# Your API definition
app = Flask(__name__)

app.debug = True
app.config['SECRET_KEY'] = 'a really really really really long secret key'

@app.route('/predict', methods=['POST'])
def predict():
    try:
            #curl --data-binary @story083838.txt -d 'tokenizer=lemma&n_gram=2-gram&threshold_factor=1.1' http:/127.0.0.1:8080/predict
            param_summarizer = {'tokenizer': 'stem',
                            'n_gram': '1-gram',
                            'threshold_factor': 1.2}
            article = [i for i in request.form.keys()][0]

            if 'tokenizer' in request.form.keys():
                param_summarizer['tokenizer'] = request.form.getlist('tokenizer')[0]
            if 'n_gram' in request.form.keys():
                param_summarizer['n_gram'] = request.form.getlist('n_gram')[0]
            if 'threshold_factor' in request.form.keys():
                param_summarizer['threshold_factor'] = float(request.form.getlist('threshold_factor')[0])
            prediction = model.run_article_summary(article, **param_summarizer)
            return jsonify({'prediction': prediction})

    except:

            return jsonify({'trace': traceback.format_exc()})

@app.route('/', methods=['get', 'post'])
def summarization():

    form = SummarizationForm()
    if not(form.threshold_factor.data):
        form.threshold_factor.data = 1.2
    if form.validate_on_submit():
        n_gram = form.n_gram.data
        tokenizer = form.tokenizer.data
        threshold_factor = form.threshold_factor.data
        text_to_summarize = form.text_to_summarize.data
        print(n_gram)
        print(tokenizer)
        print(threshold_factor)
        print(text_to_summarize)
        if text_to_summarize:
            prediction = model.run_article_summary(text_to_summarize, tokenizer, n_gram, threshold_factor)
            form.prediction.data = prediction


    return render_template('summarization.html', form=form)

if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
        port = 8080 # If you don't provide any port the port will be set to 8080
    print('model_api running')
    app.run(port=port, debug=True)