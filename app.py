from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))

@app.route('/', methods=['GET'])
def home():
    return render_template('host.html')

@app.route('/predict', methods=['POST'])
def predict(standard_to=None):
    if request.method == 'POST':
        try:
            Gender = int(request.form['Gender'])
            SeniorCitizen = int(request.form['SeniorCitizen'])
            Partner = int(request.form['Partner'])
            Dependents = int(request.form['Dependents'])
            PhoneService = int(request.form['PhoneService'])
            MultipleLines = int(request.form['MultipleLines'])
            PaperlessBilling = int(request.form['PaperlessBilling'])
            MonthlyCharges = float(request.form['MonthlyCharges'])
            TotalCharges = float(request.form['TotalCharges'])

            # Feature scaling using StandardScaler
            input_data = np.array(
                [Gender, SeniorCitizen, Partner, Dependents, PhoneService, MultipleLines, PaperlessBilling,
                 MonthlyCharges, TotalCharges]).reshape(1, -1)
            input_data_scaled = standard_to.transform(input_data)

            prediction = model.predict(input_data_scaled)
            if prediction == 1:
                return render_template('host.html', prediction_text="The customer will end the subscription")
            else:
                return render_template('host.html', prediction_text="The customer will not end the subscription")

        except Exception as e:
            print(f"Error: {e}")
            return render_template('host.html', prediction_text="Error in prediction. Please check your input data.")

if __name__ == '__main__':
    app.run(debug=True)
