import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn import preprocessing
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import matplotlib.pyplot as plt

class Model(nn.Module):

    def __init__(self, embedding_size, num_numerical_cols, output_size, layers, p=0.4):
        super().__init__()
        self.all_embeddings = nn.ModuleList([nn.Embedding(ni, nf) for ni, nf in embedding_size])
        self.embedding_dropout = nn.Dropout(p)
        self.batch_norm_num = nn.BatchNorm1d(num_numerical_cols)

        all_layers = []
        num_categorical_cols = sum((nf for ni, nf in embedding_size))
        input_size = num_categorical_cols + num_numerical_cols

        for i in layers:
            all_layers.append(nn.Linear(input_size, i))
            all_layers.append(nn.ReLU(inplace=True))
            all_layers.append(nn.BatchNorm1d(i))
            all_layers.append(nn.Dropout(p))
            input_size = i

        all_layers.append(nn.Linear(layers[-1], output_size))

        self.layers = nn.Sequential(*all_layers)

    def forward(self, x_categorical, x_numerical):
        embeddings = []
        for i,e in enumerate(self.all_embeddings):
            embeddings.append(e(x_categorical[:,i]))
        x = torch.cat(embeddings, 1)
        x = self.embedding_dropout(x)

        x_numerical = self.batch_norm_num(x_numerical)
        x = torch.cat([x, x_numerical], 1)
        x = self.layers(x)
        return x

def split_data(df):    
    msk = np.random.rand(len(df)) < .8
    train_raw = df[msk].copy()
    test_raw = df[~msk].copy()
    return train_raw, test_raw

def dataprep_ohe(df,ohe):
    numerical = list(set(df.columns.values) - set(ohe))
    temp = df[numerical].copy()
    df = pd.concat([temp,pd.get_dummies(df[ohe])], axis=1)
    print(df.columns)
    return df
    
def dataprep_stsc(df,stsc):
    ss_col = df[stsc].copy()
    scaler = preprocessing.StandardScaler().fit(ss_col.values)
    features = scaler.transform(ss_col.values)
    df[stsc] = features
    return df

def categorize_dtype(df, cols):
    for category in cols:
        df[category] = df[category].astype('category')
    return df


raw_data = pd.read_csv('Churn_Modelling.csv')
#print(raw_data.count())
#print(raw_data.nunique())

# no missing values, drop unique columns
raw_data = raw_data.drop(columns = ['Surname','CustomerId','RowNumber'])

#print(raw_data.dtypes)

outputs = ['Exited']
numerical_columns = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
ohe = ['Geography','Gender']
categorical_columns = ['Geography', 'Gender', 'HasCrCard', 'IsActiveMember']

raw_data.loc[raw_data.IsActiveMember == 0, 'IsActiveMember'] = -1
raw_data.loc[raw_data.HasCrCard == 0, 'HasCrCard'] = -1
raw_data = categorize_dtype(raw_data,categorical_columns)
raw_data = dataprep_stsc(raw_data,numerical_columns)

geo = raw_data['Geography'].cat.codes.values
gen = raw_data['Gender'].cat.codes.values
hcc = raw_data['HasCrCard'].cat.codes.values
iam = raw_data['IsActiveMember'].cat.codes.values

categorical_data = np.stack([geo, gen, hcc, iam], 1)
categorical_data = torch.tensor(categorical_data, dtype=torch.int64)

numerical_data = np.stack([raw_data[col].values for col in numerical_columns], 1)
numerical_data = torch.tensor(numerical_data, dtype=torch.float)

output = raw_data.pop('Exited')
outputs = torch.tensor(output.values).flatten()

print(categorical_data.shape)
print(numerical_data.shape)
print(outputs.shape)

categorical_column_sizes = [len(raw_data[column].cat.categories) for column in categorical_columns]
categorical_embedding_sizes = [(col_size, min(50, (col_size+1)//2)) for col_size in categorical_column_sizes]

total_records = 10000
test_records = int(total_records * .2)

categorical_train_data = categorical_data[:total_records-test_records]
categorical_test_data = categorical_data[total_records-test_records:total_records]
numerical_train_data = numerical_data[:total_records-test_records]
numerical_test_data = numerical_data[total_records-test_records:total_records]
train_outputs = outputs[:total_records-test_records]
test_outputs = outputs[total_records-test_records:total_records]

model = Model(categorical_embedding_sizes, numerical_data.shape[1], 2, [200,100,50], p=0.4)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print(model)

epochs = 300
aggregated_losses = []

for i in range(epochs):
    i += 1
    y_pred = model(categorical_train_data, numerical_train_data)
    single_loss = loss_function(y_pred, train_outputs)
    aggregated_losses.append(single_loss)

    if i%25 == 1:
        print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

    optimizer.zero_grad()
    single_loss.backward()
    optimizer.step()

print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')

plt.plot(range(epochs), aggregated_losses)
plt.ylabel('Loss')
plt.xlabel('epoch');

with torch.no_grad():
    y_val = model(categorical_test_data, numerical_test_data)
    loss = loss_function(y_val, test_outputs)
print(f'Loss: {loss:.8f}')
y_val = np.argmax(y_val, axis=1)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(confusion_matrix(test_outputs,y_val))
print(classification_report(test_outputs,y_val))
print(accuracy_score(test_outputs, y_val))































