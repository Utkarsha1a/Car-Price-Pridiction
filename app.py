# Importing essential libraries
from flask import Flask, render_template, request,redirect,url_for
import pickle
import numpy as np

# Load the Random Forest CLassifier model
filename = 'car_data.pkl'
classifier = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('new.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        name1 = str(request.form['name1'])
        print(name1)
        year = int(request.form['year'])
        tot_year = 2020 - year

        price = int(request.form['price'])
        km = int(request.form['km'])
        owner = int(request.form['owner'])
        fuel = int(request.form['fuel'])
        trans = int(request.form['trans'])
        seller = int(request.form['seller'])


        if (fuel == 0):
            fuel_P = 1
            fuel_D = 0
        elif (fuel == 1):
            fuel_P = 0
            fuel_D = 1
        else:
            fuel_P = 0
            fuel_D = 0

        if (trans == 1):
            trans_M = 1
        else:
            trans_M = 0

        if (seller == 1):
            seller_I = 1
        else:
            seller_I = 0


        data = np.array([[price, km, owner, tot_year, fuel_D, fuel_P,seller_I, trans_M]])
        print(data)
        my_prediction1 = classifier.predict(data)
        my_prediction = round(my_prediction1[0], 2)
        print(my_prediction)
        return render_template('result.html', text=my_prediction)

if __name__ == '__main__':
	app.run(debug=True)