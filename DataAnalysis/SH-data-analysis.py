#importing libraries
import pandas as pd
import numpy as np
import csv
from collections import defaultdict

disease_list = []

# cleaning the Data
def return_list(disease):
    disease_list = []
    match = disease.replace('^', '_').split('_')
    counter = 1
    for group in match:
        if counter%2 == 0:
            disease_list.append(group)
        counter += 1
    return disease_list


# Getting each row of our data table formatting the columns Getting the list of symptoms 
# for each disease
    
with open("Scraped-Data/dataset_uncleaned.csv") as csvfile:
    f_reader = csv.reader(csvfile)
    disease = ""
    weightn = 0
    disease_list = []
    dict_wt = {}
    dict_ = defaultdict(list)
    

    for row in f_reader:
        if row[0]!="\xc2\xa0" and row[0]!="":
            disease = row[0]
            disease_list = return_list(disease)
            weight = row[1]
        
        if row[2]!="\xc2\xa0" and row[2]!="":
            symptom_list = return_list(row[2])
        
            for disease in disease_list:
                for symptom in symptom_list:
                    dict_[disease].append(symptom)
                dict_wt[disease] = weight
                  
        
#     print(dict_)
                
                
# Saving the cleaned data
with open("Scraped-Data/dataset_clean.csv", "w") as csvfile:
    f_writer = csv.writer(csvfile)
    for key, values in dict_.items():
        for v in values:
            key = str.encode(key).decode('utf-8')
            f_writer.writerow([key,v,dict_wt[key]])

columns = ['Source', 'Target', 'Weight']
data = pd.read_csv("Scraped-Data/dataset_clean.csv", names=columns, encoding ="ISO-8859-1")
data.to_csv("Scraped-Data/dataset_clean.csv", index=False)

slist = []
dlist = []

with open("Scraped-Data/labeled_data.csv", "w") as csvfile:
    f_writer = csv.writer(csvfile)
    
    for key, values in dict_.items():
        for v in values:
            if v not in slist:
                f_writer.writerow([v,v,"symptom"])
                slist.append(v)
            if key not in dlist:
                f_writer.writerow([key,key,"disease"])
                dlist.append(key)
                

labeled_cols = ['Id', 'Label', 'Attribute']
labeled_data = pd.read_csv("Scraped-Data/labeled_data.csv", names=labeled_cols, encoding ="ISO-8859-1")
labeled_data.to_csv("Scraped-Data/labeled_data.csv", index=False)


# Analysing the cleaned data
data = pd.read_csv("Scraped-Data/dataset_clean.csv", encoding="ISO-8859-1")
data.to_csv("Scraped-Data/dataset_clean.csv", index=False)

print(len(data['Source'].unique()))
print(len(data['Target'].unique()))

 #Convert categorical variable into dummy/indicator variables.
df = pd.DataFrame(data)
df_1 = pd.get_dummies(df.Target)
df_s = df['Source']
df_pivoted = pd.concat([df_s, df_1], axis=1)
print(len(df_pivoted))
df_pivoted.drop_duplicates(keep='first', inplace=True)
# df_pivoted[:6]

print(len(df_pivoted))
cols = df_pivoted.columns
# print(cols[1:])
cols = cols[1:]   
df_pivoted = df_pivoted.groupby('Source').sum()
df_pivoted = df_pivoted.reset_index()
 # print(df_pivoted[:5])     
len(df_pivoted)
df_pivoted.to_csv("Scraped-Data/df_pivoted.csv")



# Defining input and target data
X = df_pivoted[cols]
y = df_pivoted['Source']


# Building the models trying out our classifiers to learn diseases from symptoms
import seaborn as sns
import matplotlib as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

test_size = 0.33
seed = 42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

# MultinomialNB
model = MultinomialNB()
model = model.fit(X_train, y_train)

# checking the model
predictions_train = model.predict(X_train)
print("MultinomialNB train score: ", accuracy_score(y_train, predictions_train))

# Evaluating the model
predictions_test = model.predict(X_test)  
print("MultinomialNB test score:", accuracy_score(y_test, predictions_test))



# Using Desicion tree
model_dt = DecisionTreeClassifier()
model_dt.fit(X_train, y_train)

predictions_train_dt = model_dt.predict(X_train)
print("DecisionTree train score:", accuracy_score(y_train, predictions_train_dt)) 

predictions_test_dt = model_dt.predict(X_test)
print("DecisionTree test score:", accuracy_score(y_test, predictions_test_dt)) 

# ''' Inferences on train and test split It can't work
#  on unseen data because it has never seen that disease 
#  before. Also, there is only one point for each disease 
#  and hence no point for this. So we need to train the 
#  model entirely. Then what will we test it on? Missing 
#  data? Say given one symptom what is the disease? 
#  This is again multilabel classification. We can work 
#  symptom on symptom. What exactly is differential 
#  diagnosis, we need to replicate that.'''


# For MNB
model_tot = MultinomialNB()
model_tot = model_tot.fit(X, y)
print("tot_MB_prediction score:", model_tot.score(X, y))

# for Decision tree model
dt_model = DecisionTreeClassifier()
model_tot_dt = DecisionTreeClassifier()
model_tot_dt = model_tot_dt.fit(X, y)
print("tot_DT_prediction score:", model_tot_dt.score(X, y))


# For MultinobialNB
disease_pred = model_tot.predict(X)
disease_real = y.values
length = (len(disease_real))


# to show predications which the current model misclassifies
for i in range(0, length):
    if disease_pred[i]!=disease_real[i]:
        print(f'Pred: {disease_pred[i]}  ACTUAL: {disease_real[i]}')
       
# for DT
disease_pred_dt = model_tot_dt.predict(X)
for i in range(0, length):
    if disease_pred_dt[i] != disease_real[i]: 
        print(f'Pred_dt: {disease_pred_dt[i]}   ACTUAL_dt: {disease_real[i]}')
        
# Visualization
from sklearn import tree
from sklearn.tree import export_graphviz 
from os import system
import pydotplus
from IPython.display import Image  

# # Create DOT data
# dot_data = export_graphviz(model_tot_dt, out_file=None, feature_names=X.columns)

# # Draw graph
# graph = pydotplus.graph_from_dot_data(dot_data)

# # Show graph
# Image(graph.create_png())

# # Create PDF
# graph.write_pdf("tree.pdf")

# # Create PNG
# graph.write_png("tree.png")

# from functions import viewDecisionTree, decisionTreeSummary
# viewDecisionTree(model_tot_dt, X.columns)

# decisionTreeSummary(model_tot_dt, X.columns)

# Analysis of the Manual data
print('\n Manual Data')
data_m = pd.read_csv("Manual-Data/Training.csv")
# print(data_m.columns)
print("column lenth", len(data_m.columns))
print("Number of diseases", len(data_m['prognosis'].unique()))

# 41 different type of target diseases are available in the manual training dataset.

df_m = pd.DataFrame(data_m)
print(len(df_m))

# The manual data contains approximately 4920 rows.
cols_m = df_m.columns
cols_m = cols_m[:-1]
print(len(cols_m))

# We have 132 symptoms in the manual data.
# setting input and target data
X_m = df_m[cols_m]
y_m = df_m['prognosis']


# Trying out our classifier to learn diseases from the symptoms

X_m_train, X_m_test, y_m_train, y_m_test = train_test_split(X_m, y_m, test_size=0.33, random_state=42)

# m_model = MultinomialNB()
# m_model = m_model.fit(X_m_train, y_m_test)

# print("MNB score1:", m_model.score(X_m_test, y_m_test))
# print("MNB score2: ", accuracy_score(X_m_test, y_m_testtes))


