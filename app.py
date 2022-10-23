from flask import Flask , render_template, request , jsonify
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict' , methods = ['POST'])
def predict():

    uid = request.form['uid']
    iid = request.form['iid']
    arr = np.array([[uid,iid]])
    arr1 = arr[0,0]
    arr2 = arr[0,1]
    prediction = model.predict(arr1,arr2)
    return render_template("index.html", prediction_text = "predict {}".format(prediction))
    
if __name__ == '__main__':
   app.run(debug=True) 
    