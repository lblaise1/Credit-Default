from flask import Flask, render_template, request
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)

@app.route("/")
@app.route('/home')
def home_page():
    return render_template('home.html', item_name = 'Phone')


@app.route('/Data')
def data_page():
    return render_template('data.html')

@app.route('/Model')
def model_page():
    return render_template('model.html')


@app.route('/Prediction')
def prediction_page():
    return render_template('prediction.html')