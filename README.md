# DEVELOPING A NEURAL NETWORK REGRESSION MODEL:

 # AIM:

To develop a neural network regression model for the given dataset.
# THEORY:

A neural network with multiple hidden layers and multiple nodes in each hidden layer is known as a deep learning system or a deep neural network. Here the basic neural network model has been created with one input layer, one hidden layer and one output layer.The number of neurons(UNITS) in each layer varies the 1st input layer has 16 units and hidden layer has 8 units and output layer has one unit.

In this basic NN Model, we have used "relu" activation function in input and hidden layer, relu(RECTIFIED LINEAR UNIT) Activation function is a piece-wise linear function that will output the input directly if it is positive and zero if it is negative.

# NEURAL NETWORK MODEL :
![Screenshot 2023-09-04 203422](https://github.com/LATHIKESHWARAN/basic-nn-model/assets/119393556/1c8945ef-7c63-45c2-a58b-1a51eac0149a)


## DESIGN STEPS :

### STEP 1 :

Loading the dataset

### STEP 2 :

Split the dataset into training and testing

### STEP 3 :

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4 :

Build the Neural Network Model and compile the model.

### STEP 5 :

Train the model with the training data.

### STEP 6 :

Plot the performance plot

### STEP 7 :

Evaluate the model with the testing data.

## PROGRAM :
NAME : LATHIKESHWARAN J

REG NO : 212222230072
```

from google.colab import auth
import gspread
from google.auth import default
import pandas as pd
import matplotlib.pyplot as plt

auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)
worksheet = gc.open('DL Data').sheet1

rows = worksheet.get_all_values()

df = pd.DataFrame(rows[1:], columns=rows[0])
df = df.astype({'Input':'float','Output':'float'})

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

df.head()

x=df[['Input']].values
x

y=df[['Output']].values
y

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=11)

Scaler=MinMaxScaler()
Scaler.fit(x_train)
Scaler.fit(x_test)
x_train1=Scaler.transform(x_train)
x_test1=Scaler.transform(x_test)
x_train1

ai_brain = Sequential([
    Dense(6,activation='relu'),
    Dense(4,activation='relu'),
    Dense(1)
])

ai_brain.compile(
    optimizer='rmsprop',
    loss='mse'
)
ai_brain.fit(x_train1,y_train,epochs=4000)

loss_df=pd.DataFrame(ai_brain.history.history)
loss_df.plot()
plt.title('Training Loss Vs Iteration Plot')

ai_brain.evaluate(x_test1,y_test)

x_n1=[[66]]
x_n1_1=Scaler.transform(x_n1)
ai_brain.predict(x_n1_1)

```
# Dataset Information :


![Screenshot 2023-09-04 203723](https://github.com/LATHIKESHWARAN/basic-nn-model/assets/119393556/85fb1e40-8589-470d-8419-c77a8598f865)


## OUTPUT :


### Training Loss Vs Iteration Plot :

![Screenshot 2023-09-04 203808](https://github.com/LATHIKESHWARAN/basic-nn-model/assets/119393556/1a09eea8-15c7-484b-816a-dfbff348ac86)



### Test Data Root Mean Squared Error :
![Screenshot 2023-09-04 203820](https://github.com/LATHIKESHWARAN/basic-nn-model/assets/119393556/5a89257b-94f1-4fac-9ec1-1d91a26a5f20)


### New Sample Data Prediction :

![Screenshot 2023-09-04 203904](https://github.com/LATHIKESHWARAN/basic-nn-model/assets/119393556/3a4ee154-cd7a-4a83-9a6a-e6c710c84b4b)


## RESULT :
Thus a neural network regression model for the given dataset is written and executed successfully
