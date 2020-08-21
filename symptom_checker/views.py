#importing libraries
import pickle
# import DataAnalysis.predictionChatbot as model
# from DataAnalysis.predictionChatbot import getSeverityDict, getDescription, getprecautionDict, runAll

import pandas as pd
import pyttsx3
import joblib
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier,_tree
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import csv
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


from django.shortcuts import render
from django.http import JsonResponse
from django.http import HttpResponse

# Create your views here
dict_file = open("DataAnalysis/symptom_dictionary.pkl", "rb")
sym_dict = pickle.load(dict_file)
global_symp = 'symptom'
# symptom = '';
res = {}
numberOf_days = 0
symptoms_given = []
conf_inputx = 0


res["res1"] = "Searches related to input for: "
res["sym_num"] = 0
res["symp_tom"] = global_symp

def Show_symptom_page(request):       
    return render(request, 'symptom_checker/symptoms.html',  {'sym_dict':sym_dict})


def getPrediction(request):
    global global_symp
    if request.is_ajax() and request.method == 'GET':
        global_symp = request.GET.get('symptom')
        response = {}
        response["text"] = global_symp
        return JsonResponse(response, status=200)
    
def specific_symptom(request):
    return render(request, 'symptom_checker/specific_checker.html', {'sym': global_symp, 'res': res})

# def testModel(request):
#     global numberOf_days
#     if request.is_ajax() and request.method == 'GET':
#         numberOf_days = request.GET.get('numOfDays')
#         return JsonResponse(res)
    
def getNumOfDays(request):
    if request.is_ajax() and request.method == 'GET':
        print("before", numberOf_days)
        runAll()
        return JsonResponse(res)
    


# running python code
training = pd.read_csv('DataAnalysis/SH-chatbot-Data/Training.csv')
testing= pd.read_csv('DataAnalysis/SH-chatbot-Data/Testing.csv')
cols= training.columns
cols= cols[:-1]
feature_names = cols
x = training[cols]
y = training['prognosis']
y1= y

# grouping the training set by the maximum occuring disease
reduced_data = training.groupby(training['prognosis']).max()

#mapping strings to numbers
le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)

# splitting out data into train and test variables
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
testx    = testing[cols]
testy    = testing['prognosis']  
testy    = le.transform(testy)

# trying out the decision try model
# clf1  = DecisionTreeClassifier()
# clf = clf1.fit(x_train,y_train)

joblib_file = "DataAnalysis/prediction_dt.pkl"
# Load from file
tree = joblib.load(joblib_file)

# calculating feature importances
importances = tree.feature_importances_
indices = np.argsort(importances)[::-1]
features = cols

severityDictionary=dict()
description_list = dict()
precautionDictionary=dict()

symptoms_dict = {}

for index, symptom in enumerate(x):
       symptoms_dict[symptom] = index
       
def calc_condition(exp,days):
    sum=0
    for item in exp:
         sum=sum+severityDictionary[item]
         print(sum)
         print(len(exp))
    if((int((sum*days)) > 13)):
        print("You should take the consultation from doctor. ")
    else:
        print("It might not be that bad but you should take precautions.")
        
def getDescription():
    global description_list
    with open('DataAnalysis/SH-chatbot-Data/symptom_Description.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            _description={row[0]:row[1]}
            description_list.update(_description)
            

def getSeverityDict():
    global severityDictionary    
    with open('DataAnalysis/SH-chatbot-Data/symptom_severity.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count= 0
        for row in csv_reader:
            if(row):
                _diction={row[0]:int(row[1])}
                severityDictionary.update(_diction)
        
def getprecautionDict():
    global precautionDictionary
    with open('DataAnalysis/SH-chatbot-Data/symptom_precaution.csv') as csv_file:

        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            _prec={row[0]:[row[1],row[2],row[3],row[4]]}
            precautionDictionary.update(_prec)
        

def check_pattern(dis_list,inp):
    import re
    pred_list=[]
    ptr=0
    inp = inp.lower()
    res = len(inp.split())
    if res > 1:
        inp = inp.replace(" ", "_")
    
    patt = "^" + inp + "$"
    regexp = re.compile(inp)
    for item in dis_list:

#         print(f"comparing {inp} to {item}")
        if regexp.search(item):
            pred_list.append(item)
            # return 1,item
    if(len(pred_list)>0):
        return 1,pred_list
    else:
        return ptr,item


def sec_predict(symptoms_exp):
    df = pd.read_csv('DataAnalysis/SH-chatbot-Data/Training.csv')
    X = df.iloc[:, :-1]
    y = df['prognosis']
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)
    # rf_clf = DecisionTreeClassifier()
    # rf_clf.fit(X_train, y_train)
    joblib_newfile = "DataAnalysis/prediction_model.pkl"
    # Load from file
    joblib_model = joblib.load(joblib_newfile)
    
    
    
    symptoms_dict = {}

    for index, symptom in enumerate(X):
        symptoms_dict[symptom] = index

    input_vector = np.zeros(len(symptoms_dict))
    for item in symptoms_exp:
        input_vector[[symptoms_dict[item]]] = 1
    return joblib_model.predict([input_vector])

def print_disease(node):
    #print(node)
    node = node[0]
    #print(len(node))
    val  = node.nonzero() 
    # print(val)
    disease = le.inverse_transform(val[0])
    return disease


def tree_to_code(request):
    global res
    global numberOf_days    
    getSeverityDict()
    getDescription()
    getprecautionDict()
    
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    chk_dis=",".join(feature_names).split(",")
    symptoms_present = []


    # conf_inp=int()
    while True:
        if global_symp != "symptom":
            disease_input = global_symp
            conf,cnf_dis = check_pattern(chk_dis,disease_input)
            if conf == 1:
                for num,it in enumerate(cnf_dis):
                    res["sym_num"] = num
                    res["symp_tom"] = it
                if num!=0:
                    res["select_one_u_mearnt"] = "Select the one you meant:  " + (0 - {num})
                    # print(f"Select the one you meant (0 - {num}):  ", end="")
                    conf_inp = conf_inputx
                else:
                    conf_inp=0

                disease_input = cnf_dis[conf_inp]
                break
                # print("Did you mean: ",cnf_dis,"?(yes/no) :",end="")
                # conf_inp = input("")
                # if(conf_inp=="yes"):
                #     break
                
            else:
                res["no_symptom"] = "Enter valid symptom."
                # print("Enter valid symptom.")
                
        
    if request.is_ajax() and request.method == 'POST':
        symptoms_givenNew  = []
        numberOf_days = request.POST.get('numOfDays')
     
        
        num_days = numberOf_days
        print(numberOf_days)
        res['numberOf_days_success'] = "numberOf_days_success"   
        
        if numberOf_days != None:
           
            def recurse(node, depth):
                global symptoms_givenNew
                indent = "  " * depth
                if tree_.feature[node] != _tree.TREE_UNDEFINED:
                    name = feature_name[node]
                    threshold = tree_.threshold[node]

                    if name == disease_input:
                        val = 1
                    else:
                        val = 0
                    if  val <= threshold:
                        recurse(tree_.children_left[node], depth + 1)
                    else:
                        symptoms_present.append(name)
                        recurse(tree_.children_right[node], depth + 1)
                else:
                    present_disease = print_disease(tree_.value[node])
                    # print( "You may have " +  present_disease )
                    red_cols = reduced_data.columns 
                    symptoms_given = red_cols[reduced_data.loc[present_disease].values[0].nonzero()]
                    
                    symptoms_givenNew = list(symptoms_given)
                    res['symptoms_given'] = symptoms_givenNew
                    symptoms_exp = []
                    symptoms_exp.append(disease_input)
                    
                    second_prediction=sec_predict(symptoms_exp)
                    res['second_prediction'] = str(second_prediction)
                    print(second_prediction)
                    calc_condition(symptoms_exp,int(numberOf_days))
                    if(present_disease[0] == second_prediction[0]):
                        
                        res['youMayHave'] = str(present_disease[0])
                        
                        
                        res["desc"] = str(description_list[present_disease[0]])
    
                        
                    else:
                        res['youMayHave1'] = str(present_disease[0])
                        res['youMayHave2'] = str(second_prediction).replace("[", " ").replace("'", " ").replace("]", " ").strip()
                        print("You may have ", present_disease[0], "or ", second_prediction[0])
                        res["desc1"] =  str(description_list[present_disease[0]])
                        res["desc2"] = str(description_list[second_prediction[0]])
                        
                        
                        
                    precution_list = list(precautionDictionary[present_disease[0]])
                    
                    
                    for  i,j in enumerate(precution_list):
                        res["precaution"] = (precution_list)
                        print(i+1,")",j)    
                
        recurse(0,1) 
        
    
        return JsonResponse(res)
    # else:
    #     res['numberOf_days_error'] = "Enter number of days."
    #     return JsonResponse(res)
        
    
        
        
       
            
    

            
    
    
    

    
    
    
    


def runAll():
    # getSeverityDict()
    # getDescription()
    # getprecautionDict()
    # getInfo()
    tree_to_code()
    

     
