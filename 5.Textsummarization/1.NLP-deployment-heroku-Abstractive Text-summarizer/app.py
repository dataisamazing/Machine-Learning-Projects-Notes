from nltk.corpus import stopwords
import numpy as np
import networkx as nx
import regex
from flask import Flask, request, jsonify, render_template
import nltk
import textwrap
from transformers import PegasusForConditionalGeneration, PegasusTokenizer


def generate_summary(text):

    # Create tokenizer
    tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-xsum")
    # load pretrained model
    model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum")

    # convert into tokens (number representation of text)   
    tokens = tokenizer(text, truncation=True, padding="longest", return_tensors="pt")
    summary = model.generate(**tokens)
    #Summarized = wrapper.fill(tokenizer.decode(summary[0])).strip()
    Summarized = tokenizer.decode(summary[0])
    return Summarized

#----------FLASK-----------------------------#

app = Flask(__name__)
@app.route('/templates', methods =['POST'])
def original_text_form():
		text = request.form['input_text']
		summary = generate_summary(text)
		return render_template('index1.html', title = "Summarizer", original_text = text, output_summary = summary, num_sentences = 5)

@app.route('/')
def homepage():
	title = "TEXT summarizer"
	return render_template('index1.html', title = title)

if __name__ == "__main__":
	app.debug = True
	app.run()
