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


# @app.route('/Prediction')
# def prediction_page():
#     return render_template('prediction.html')
@app.route("/Prediction")
def submit():
    loan_amnt = request.form.get("loan_amnt")
    inq_last_6mths = request.form.get("inq_last_6mths")
    revol_util = request.form.get("revol_util")
    annual_inc = request.form.get("annual_inc")
    purpose = request.form.get("purpose")
    pub_rec_bankruptcies = request.form.get("pub_rec_bankruptcies")
    # or not revol_util or not annual_inc or not pub_rec_bankruptcies
    # if not loan_amnt or not inq_last_6mths:
    #     return "All fields must be filled out"

    return render_template("pred.html")

if __name__ == "__main__":
    app.run(debug=True)