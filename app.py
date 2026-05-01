from flask import Flask, render_template, request
from flask_sqlalchemy import SQLAlchemy
import numpy as np
import pickle
import os

app = Flask(__name__)

# ---------- Load model artifacts at startup ----------
ARTIFACTS_PATH = 'model_artifacts.pkl'

if os.path.exists(ARTIFACTS_PATH):
    with open(ARTIFACTS_PATH, 'rb') as f:
        artifacts = pickle.load(f)
    MODEL_LOADED = True
    print(f'[OK] Loaded model artifacts from {ARTIFACTS_PATH}')
else:
    artifacts = None
    MODEL_LOADED = False
    print(f'[WARN] {ARTIFACTS_PATH} not found. Run `python train_model.py` first.')


# ---------- Routes ----------
@app.route("/")
@app.route('/home')
def home_page():
    return render_template('home.html', item_name='Phone')


@app.route('/Data')
def data_page():
    return render_template('data.html')


@app.route('/Model')
def model_page():
    return render_template('model.html')


@app.route("/Prediction", methods=['GET', 'POST'])
def submit():
    # GET request: show the empty form. We pass the available purpose
    # categories so the template can render a dropdown if it wants to.
    if request.method == 'GET':
        purposes = artifacts['purpose_categories'] if MODEL_LOADED else []
        return render_template('pred.html',
                               purposes=purposes,
                               result=None,
                               model_loaded=MODEL_LOADED)

    # POST request: gather form fields
    try:
        loan_amnt = float(request.form.get("loan_amnt"))
        inq_last_6mths = float(request.form.get("inq_last_6mths"))
        revol_util = float(request.form.get("revol_util"))
        annual_inc = float(request.form.get("annual_inc"))
        purpose = request.form.get("purpose")
        pub_rec_bankruptcies = float(request.form.get("pub_rec_bankruptcies"))
    except (TypeError, ValueError):
        return render_template('pred.html',
                               purposes=artifacts['purpose_categories'] if MODEL_LOADED else [],
                               result={'error': 'Please fill in all fields with valid numbers.'},
                               model_loaded=MODEL_LOADED)

    if not MODEL_LOADED:
        return render_template('pred.html',
                               purposes=[],
                               result={'error': 'Model not trained yet. Run `python train_model.py` first.'},
                               model_loaded=False)

    # Training script used ln_annual_inc, so we need to log-transform here too
    ln_annual_inc = np.log(annual_inc) if annual_inc > 0 else 0.0

    # Build feature vector the SAME way training did
    purpose_label_enc = artifacts['purpose_label_enc']
    purpose_onehot_enc = artifacts['purpose_onehot_enc']
    scaler = artifacts['scaler']
    n_cat_cols = artifacts['n_cat_cols']

    # If user picked a purpose the model never saw, fall back to first known one
    if purpose not in artifacts['purpose_categories']:
        purpose = artifacts['purpose_categories'][0]

    encoded_label = purpose_label_enc.transform([purpose])
    cat_part = purpose_onehot_enc.transform(encoded_label.reshape(-1, 1)).toarray()

    num_part = np.array([[loan_amnt, inq_last_6mths, revol_util,
                          pub_rec_bankruptcies, ln_annual_inc]])

    features = np.concatenate([cat_part, num_part], axis=1)
    features[:, n_cat_cols:] = scaler.transform(features[:, n_cat_cols:])

    # Predict
    model = artifacts['model']
    probs = model.predict_proba(features)[0]
    default_prob = float(probs[1])  # class 1 = default
    threshold = artifacts['threshold']
    will_default = default_prob > threshold

    # Risk-factor breakdown: simple, transparent rules of thumb so the user
    # sees WHY the model rated them this way. These are heuristics, not the
    # model's true coefficients, but they're consistent with how the model
    # was trained.
    risk_factors = []

    if revol_util >= 70:
        risk_factors.append(('Revolving utilization', 'High',
                             f'{revol_util:.0f}% utilization signals heavy reliance on revolving credit.'))
    elif revol_util >= 40:
        risk_factors.append(('Revolving utilization', 'Moderate',
                             f'{revol_util:.0f}% utilization is in the cautious range.'))
    else:
        risk_factors.append(('Revolving utilization', 'Low',
                             f'{revol_util:.0f}% utilization is healthy.'))

    if inq_last_6mths >= 3:
        risk_factors.append(('Recent credit inquiries', 'High',
                             f'{int(inq_last_6mths)} inquiries in 6 months suggests credit-seeking behavior.'))
    elif inq_last_6mths >= 1:
        risk_factors.append(('Recent credit inquiries', 'Moderate',
                             f'{int(inq_last_6mths)} inquiries in 6 months is normal but worth noting.'))
    else:
        risk_factors.append(('Recent credit inquiries', 'Low',
                             'No recent credit inquiries.'))

    if pub_rec_bankruptcies >= 1:
        risk_factors.append(('Public bankruptcies', 'High',
                             f'{int(pub_rec_bankruptcies)} prior bankruptcy filing(s) on record.'))
    else:
        risk_factors.append(('Public bankruptcies', 'Low',
                             'No prior bankruptcies on record.'))

    loan_to_income = loan_amnt / annual_inc if annual_inc > 0 else 0
    if loan_to_income >= 0.5:
        risk_factors.append(('Loan-to-income ratio', 'High',
                             f'Loan is {loan_to_income*100:.0f}% of annual income.'))
    elif loan_to_income >= 0.25:
        risk_factors.append(('Loan-to-income ratio', 'Moderate',
                             f'Loan is {loan_to_income*100:.0f}% of annual income.'))
    else:
        risk_factors.append(('Loan-to-income ratio', 'Low',
                             f'Loan is {loan_to_income*100:.0f}% of annual income.'))

    # Verdict label
    if default_prob >= 0.50:
        verdict = 'High Risk — Likely to Default'
        verdict_class = 'verdict-high'
    elif default_prob >= threshold:
        verdict = 'Elevated Risk — Caution Advised'
        verdict_class = 'verdict-medium'
    else:
        verdict = 'Low Risk — Likely to Repay'
        verdict_class = 'verdict-low'

    result = {
        'default_probability': round(default_prob * 100, 1),
        'repay_probability': round((1 - default_prob) * 100, 1),
        'verdict': verdict,
        'verdict_class': verdict_class,
        'will_default': will_default,
        'threshold': round(threshold * 100, 1),
        'risk_factors': risk_factors,
        'inputs': {
            'Loan Amount': f'${loan_amnt:,.0f}',
            'Annual Income': f'${annual_inc:,.0f}',
            'Loan Purpose': purpose,
            'Revolving Utilization': f'{revol_util:.0f}%',
            'Recent Inquiries (6mo)': int(inq_last_6mths),
            'Public Bankruptcies': int(pub_rec_bankruptcies),
        },
    }

    return render_template('pred.html',
                           purposes=artifacts['purpose_categories'],
                           result=result,
                           model_loaded=True)


if __name__ == "__main__":
    app.run(port=5001, debug=True)
