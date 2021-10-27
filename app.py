from flask import Flask,render_template,request
import pickle
from sklearn.preprocessing import StandardScaler
from flask_cors import CORS,cross_origin
import pandas as pd

app =Flask(__name__)

@app.route('/',methods=['Get'])
@cross_origin()
def homepage():
    return render_template('index.html')

@app.route('/predict',methods=['Get','Post'])
@cross_origin()
def index():
    if request.method == 'POST':
        try:
            Processtemperature = float(request.form['Process_temperature_[K]'])
            Rotationalspeed = float(request.form['Rotational_speed_[rpm]'])
            Torque = float(request.form['Torque_[Nm]'])
            Toolwear = float(request.form['Tool_wear_[min]'])
            TWF = float(request.form['TWF'])
            HDF = float(request.form['HDF'])
            PWF = float(request.form['PWF'])
            OSF = float(request.form['OSF'])
            RNF = float(request.form['RNF'])
            filename = 'linear_regression.sav'
            load_model = pickle.load(open(filename,'rb'))
            scaler = StandardScaler()
            df = pd.read_csv('ai4i2020.csv')
            y = df[['Air temperature [K]']]
            x = df.drop(columns=['UDI', 'Product ID', 'Type', 'Air temperature [K]','Machine failure'])
            arr = scaler.fit_transform(x)
            trm = scaler.transform([[Processtemperature,Rotationalspeed,Torque,Toolwear,TWF,HDF,PWF,OSF,RNF]])
            prediction=load_model.predict(trm)
            print('Prediction is ',prediction)
            return render_template('results.html',prediction=prediction[0][0])

        except Exception as e:
            print('The Exception message is: ', e)
            return 'something is wrong'

    else:
        return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)

