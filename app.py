from flask import Flask, render_template, request, url_for
import pickle
import numpy as np 

app = Flask(__name__)

model = pickle.load(open('saved_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('test.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():

    print("AA")
    int_features = int(request.form.get('Marks'))
    print(int_features)
    int_features = np.array(int_features)
    final_features = (int_features).reshape(1,-1)
    prediction = model.predict(final_features)

    output = round(prediction[0])

    return render_template('test.html', prediction_text='Your Predicted AIR is {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)
    