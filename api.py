from flask import Flask, render_template, request
from ml import SlidingWindowTransform, NeuralNetwork, RNeuralNetwork, OLS
import csv
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pandas.tseries.holiday import USFederalHolidayCalendar
import copy
import numpy as np
import csv
import random
import pickle
import matplotlib
import matplotlib.pyplot as plt
import io
import base64
app = Flask(__name__)
matplotlib.use('Agg')

@app.route("/")
def home():
    return render_template('website.html')
@app.route("/getData", methods=['GET'])
def runModels():
    data = predict(int(request.args.get("months")))
    olddata = getOldData()
    fig,ax = plt.subplots()
    fig.subplots_adjust(bottom=.600)

    ax.plot(olddata[:,0], olddata[:,1])
    times = [datetime(2022, 1, 1) + timedelta(days=i) for i in range(len(data))]
    ax.plot(times, data)
    b = io.BytesIO()
    plt.savefig(b, format="png")
    b.seek(0)
    b = base64.b64encode(b.read()).decode()
    resp = {}
    resp["image"] = b
    resp["total"] = np.sum(data)
    return resp
def getOldData():
    with open("data_daily.csv") as data_file:
        reader = csv.reader(data_file)
        next(reader)
        data = []
        for i in reader:
            data.append(i)
        data = [[datetime.strptime(i[0],"%Y-%m-%d") , int(i[1])] for i in data]
    return np.array(data)
def getData():
    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start='2021-01-01', end='2022-12-31').to_pydatetime()
    # Read Data from file
    with open("data_daily.csv") as data_file:
        reader = csv.reader(data_file)
        next(reader)
        data = []
        for i in reader:
            data.append(i)
        data = [[datetime.strptime(i[0],"%Y-%m-%d") , int(i[1])] for i in data]
        scaler = MinMaxScaler((-1,1))
        data = np.array(data)
        scaler.fit(data[:,1].reshape(-1,1))
        pickle.dump(scaler, open("scaler.pkl", "wb"))

        olddata = copy.deepcopy(data[:,1])
        data[:,1] = scaler.transform(data[:,1].reshape(-1,1)).flatten()
        # Remove Trend from data
        data[:,1] = data[:,1] - np.append([0],data[:,1][:-1])
        data = data[1:,:]
        print(data)
        # Transform data and add Features
        df = SlidingWindowTransform(8).transform(data)
        Y = df.iloc[:,1:].to_numpy()
        X= df.iloc[:,:df.shape[1]-1].to_numpy()
        newX = np.zeros((X.shape[0], X.shape[1], 5))
        for i in range(len(X)):
            for j in range(len(X[i])):
                newX[i,j,:] = [X[i,j], np.mean(data[:i+j+1,1]), np.std(data[:i+j+1,1]), int(data[i+j,0] in holidays), data[i+j,0].weekday()  ]
        return newX, Y
def createModel():
        olddata = getOldData()
        newX, Y = getData()
        scaler = pickle.load(open("scaler.pkl","rb"))
        olddata[:,1] = scaler.transform(olddata[:,1].reshape(-1,1)).flatten()

        Y = Y.reshape((Y.shape[0],Y.shape[1],1))
        # Train Recurrent NN
        nn = RNeuralNetwork()
        nn.fit(newX,Y)
        linear_model = OLS()
        linear_model.fit(olddata,olddata[:,1].reshape(-1,1))
        nn.save()
        linear_model.save()

def predict(months):
    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start='2021-01-01', end='2022-12-31').to_pydatetime()
    newX,Y = getData()
    data = getOldData()
    last = data[-1,1]
    scaler = pickle.load(open("scaler.pkl","rb"))
    data[:,1] = scaler.transform(data[:,1].reshape(-1,1)).flatten()

    nn = RNeuralNetwork()
    nn.load()
    linear_model = OLS()
    linear_model.load()
    new_rows = []
    new = newX[-1, :].reshape(1,newX.shape[1],5)
    past2 = data[:,1]
    days = [31,28,31,30,31,30,31,31,30,31,30,31]
    jan = datetime(2022,1,1)
    for i in range(sum(days[:months])):
        new = new.reshape(1,7,5) 
        for i in range(len(new[0])):
            new[0,i,0] += random.uniform(-0.05,.05)
        res = nn.predict(new)
        np.append(past2, [res[-1][-1]])
        new = np.append(new[0,1:], [[res[-1][-1] , np.mean(past2 ), np.std(past2),  int(jan in holidays), jan.weekday()]], axis=0)
        jan += timedelta(days=1)
        new_rows.append(res[-1])
    B = linear_model.weights
    lpoints = np.array([B[0,0]+i*B[1,0]+ (i**2)*B[2,0] + (i**3)*B[3,0] for i in range(365, 365+len(new_rows))]).flatten()
    lpoints = lpoints
    n = scaler.inverse_transform(lpoints.reshape(-1,1) + np.array([i[-1] for i in new_rows]).reshape(-1,1)).flatten()
    return n
if __name__ == "__main__":
    createModel()
    app.run(host='0.0.0.0')
