from flask import Flask, render_template, url_for, request
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')    

def predict(dist, season, crop, area):
    df = dataset.loc[dataset['District_Name'] == dist]
    s = df.iloc[:, 3:-1].values
    s = s.mean(axis=0)
    
    z = dataset.iloc[0, :-1].values
    z = z.reshape(1, -1)
    z[0][0] = dist
    z[0][1] = season
    z[0][2] = crop
    for i in range(len(s)):
        z[0][i+3] = s[i]
        
    z[:, 0] = labelencoder0.transform(z[:, 0])
    z[:, 1] = labelencoder1.transform(z[:, 1])
    z[:, 2] = labelencoder2.transform(z[:, 2])
    
    z = ct.transform(z)
    
   
    z = np.array(z[0][1:])
    z = z.reshape(1, -1)
    
    z_pred = round(regressor.predict(z)[0] * area, 2)

    return z_pred

def generateGraph(dist, season, area):
    plt.close('all')
    df = dataset.loc[dataset['District_Name'] == dist]
    df = df.loc[dataset['Season'] == season]
    crops = df.Crop.unique()

    O = []
    P = []
    for c in crops:
        if((c != 'Sugarcane' and c != 'Cotton(lint)') or season == 'Whole Year'):
            O.append(c)
            P.append(predict(dist, season, c, area))
    return O, P

@app.route('/predict', methods=['POST'])
def result():
    if request.method == 'POST':
        # Input
        dist = request.form['district']
        season = request.form['season']
        crop = request.form['crop']
        area = float(request.form['area'])
       
        z_pred = predict(dist, season, crop, area);
       
        O, P = generateGraph(dist, season, area)
        
        m1, m2 = (list(t) for t in zip(*sorted(zip(P, O))))
        if(len(m1) >= 3):
            m1 = m1[-3:]
            m2 = m2[-3:]

    return render_template('result.html', prediction = str(z_pred), crop = O, pred = P, m1 = m1, m2 = m2, c = crop)

if __name__ == '__main__':
    # Importing the dataset
    dataset = pd.read_excel('Final_dataset_backup.xlsx')
    
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values
    
    # Encoding categorical data
    
    labelencoder0 = LabelEncoder()
    X[:, 0] = labelencoder0.fit_transform(X[:, 0])
    
    labelencoder1 = LabelEncoder()
    X[:, 1] = labelencoder1.fit_transform(X[:, 1])

    labelencoder2 = LabelEncoder()
    X[:, 2] = labelencoder2.fit_transform(X[:, 2])
    
    ct = ColumnTransformer([('one_hot_encoder', OneHotEncoder(handle_unknown = 'ignore'),[0,1,2])],remainder='passthrough')   
    
    X = ct.fit_transform(X)
    
    
    
    
    # Loading saved model
    from sklearn.externals import joblib
    regressor = joblib.load('model.sav')
    app.run(debug=True)