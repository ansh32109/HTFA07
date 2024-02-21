from utils.types import AnonMethod
import os
import sys
import argparse
import numpy as np
import pandas as pd
from metrics import NCP, DM, CAVG

from algorithms import (
        k_anonymize,
        read_tree)
from datasets import get_dataset_params
from utils.data import read_raw, write_anon, numberize_categories


# In[52]:


import warnings
import os
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, f1_score, accuracy_score, recall_score, precision_score
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import matplotlib.patches as mpatches
import pickle


# In[51]:


import random

class Anonymizer:
    def __init__(self, args):
        self.method = args[0]
        assert self.method in ["mondrian", "topdown", "cluster", "mondrian_ldiv", "classic_mondrian", "datafly"]
        self.k = args[1]
        self.data_name = args[2]
        self.csv_path = args[2]+'.csv'

        # Data path
        self.path = os.path.join('data', args[2])  # trailing /

        # Dataset path
        self.data_path = os.path.join(self.path, self.csv_path)

        # Generalization hierarchies path
        self.gen_path = os.path.join(
            self.path,
            'hierarchies')  # trailing /

        # folder for all results
        res_folder = os.path.join(
            'results', 
            args[2], 
            self.method)

        # path for anonymized datasets
        self.anon_folder = res_folder  # trailing /
        
        os.makedirs(self.anon_folder, exist_ok=True)

    def anonymize(self):
        data = pd.read_csv(self.data_path, delimiter=';')
        ATT_NAMES = list(data.columns)
        
        data_params = get_dataset_params(self.data_name)
        QI_INDEX = data_params['qi_index']
        IS_CAT2 = data_params['is_category']

        QI_NAMES = list(np.array(ATT_NAMES)[QI_INDEX])
        IS_CAT = [True] * len(QI_INDEX) # is all cat because all hierarchies are provided
        SA_INDEX = [index for index in range(len(ATT_NAMES)) if index not in QI_INDEX]
        SA_var = [ATT_NAMES[i] for i in SA_INDEX]

        ATT_TREES = read_tree(
            self.gen_path, 
            self.data_name, 
            ATT_NAMES, 
            QI_INDEX, IS_CAT)

        raw_data, header = read_raw(
            self.path, 
            self.data_name, 
            QI_INDEX, IS_CAT)

        anon_params = {
            "name" :self.method,
            "att_trees" :ATT_TREES,
            "value" :self.k,
            "qi_index" :QI_INDEX, 
            "sa_index" :SA_INDEX
        }

        if self.method == AnonMethod.CLASSIC_MONDRIAN:
            mapping_dict,raw_data = numberize_categories(raw_data, QI_INDEX, SA_INDEX, IS_CAT2)
            anon_params.update({'mapping_dict': mapping_dict})
            anon_params.update({'is_cat': IS_CAT2})

        if self.method == AnonMethod.DATAFLY:
            anon_params.update({
                'qi_names': QI_NAMES,
                'csv_path': self.data_path,
                'data_name': self.data_name,
                'dgh_folder': self.gen_path,
                'res_folder': self.anon_folder})

        anon_params.update({'data': raw_data})

        print(f"Anonymize with {self.method}")
        anon_data, runtime = k_anonymize(anon_params)

        # Write anonymized table
        if anon_data is not None:
            nodes_count = write_anon(
                self.anon_folder, 
                anon_data, 
                header, 
                self.k, 
                self.data_name)

        if self.method == AnonMethod.CLASSIC_MONDRIAN:
            ncp_score, runtime = runtime
        else:
            # Normalized Certainty Penalty
            ncp = NCP(anon_data, QI_INDEX, ATT_TREES)
            ncp_score = ncp.compute_score()

        # Discernibility Metric

        raw_dm = DM(raw_data, QI_INDEX, self.k)
        raw_dm_score = raw_dm.compute_score()

        anon_dm = DM(anon_data, QI_INDEX, self.k)
        anon_dm_score = anon_dm.compute_score()

        # Average Equivalence Class

        raw_cavg = CAVG(raw_data, QI_INDEX, self.k)
        raw_cavg_score = raw_cavg.compute_score()

        anon_cavg = CAVG(anon_data, QI_INDEX, self.k)
        anon_cavg_score = anon_cavg.compute_score()
        
        print(f"The K value in this iteration is:{self.k}")
        print(f"NCP score (lower is better): {ncp_score:.3f}")
        print(f"CAVG score (near 1 is better): BEFORE: {raw_cavg_score:.3f} || AFTER: {anon_cavg_score:.3f}")
        print(f"DM score (lower is better): BEFORE: {raw_dm_score} || AFTER: {anon_dm_score}")
        print(f"Time execution: {runtime:.3f}s")

        return ncp_score, raw_cavg_score, anon_cavg_score, raw_dm_score, anon_dm_score


# def main():
# #     args = []
# #     args.append(input("Input the method: "))
# #     args.append(int(input("Input the k: ")))
# #     args.append(input("Input the dataset: "))
    
    

# if __name__ == '__main__':
#     main()


# In[71]:


class StandardModel:
    def __init__(self):
        pass
        
        
    def standardmodel(self):
        df_data= pd.read_csv("./data/adult/adult.csv", delimiter=';')
        df_data["native-country"].value_counts().index[0]
        col_names = df_data.columns
        print(col_names)
        for c in col_names: 
            df_data = df_data.replace("?", np.NaN) 
        df_data = df_data.apply(lambda x:x.fillna(x.value_counts().index[0]))
        category_col =['workclass', 'education', 'marital-status', 'occupation', 
               'race', 'sex', 'native-country', 'salary-class']  
        labelEncoder = preprocessing.LabelEncoder() 
  
        mapping_dict ={} 
        for col in category_col: 
            df_data[col] = labelEncoder.fit_transform(df_data[col]) 
  
            le_name_mapping = dict(zip(labelEncoder.classes_, 
                        labelEncoder.transform(labelEncoder.classes_))) 
  
            mapping_dict[col]= le_name_mapping 
        print(mapping_dict) 
        Y = df_data['salary-class']
        X = df_data.drop('salary-class', axis = 1)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 100) 
        tsne = TSNE(n_components=2, random_state=0)
        X_tsne = tsne.fit_transform(X.values)
        
        blue_patch = mpatches.Patch(color='#0A0AFF', label='<=50K')

        red_patch = mpatches.Patch(color='#AF0000', label='>50K')

        plt.figure(figsize=(20,10))

        plt.scatter(X_tsne[:,0], X_tsne[:,1], c=(Y == 0), cmap='coolwarm', label='<=50K', linewidths=2)

        plt.scatter(X_tsne[:,0], X_tsne[:,1], c=(Y == 1), cmap='coolwarm', label='>50K', linewidths=2)

        plt.title('t-SNE')

        plt.legend(handles=[blue_patch, red_patch])
        
        
        
        rf = RandomForestClassifier(n_estimators=100, random_state=0)
        rf.fit(X_train, y_train)
        ypredrf = rf.predict(X_test)
        print(confusion_matrix(y_test, ypredrf))
        print(classification_report(y_test, ypredrf))
        print("Accuracy Score:", accuracy_score(y_test, ypredrf))
        print("Recall Score:", recall_score(y_test, ypredrf))
        print("Precision Score:", precision_score(y_test, ypredrf))
        print("ROC AUC Score: ", roc_auc_score(y_test, ypredrf))
        rf_fp, rf_tp, rf_threshold = roc_curve(y_test, ypredrf)
        print("Threshold:", rf_threshold)
        
        
        gbc = GradientBoostingClassifier(n_estimators=100, random_state=0)
        gbc.fit(X_train, y_train)
        ypredgbc = gbc.predict(X_test)
        print(confusion_matrix(y_test, ypredgbc))
        print(classification_report(y_test, ypredgbc))
        print("Accuracy Score:", accuracy_score(y_test, ypredgbc))
        print("Recall Score:", recall_score(y_test, ypredgbc))
        print("Precision Score:", precision_score(y_test, ypredgbc))
        print("ROC AUC Score: ", roc_auc_score(y_test, ypredgbc))
        gbc_fp, gbc_tp, gbc_threshold = roc_curve(y_test, ypredgbc)
        print("Threshold:", gbc_threshold)
        
        
        abc = AdaBoostClassifier(n_estimators=100, random_state=0)
        abc.fit(X_train, y_train)
        ypredabc = abc.predict(X_test)
        print(confusion_matrix(y_test, ypredabc))
        print(classification_report(y_test, ypredabc))
        print("Accuracy Score:", accuracy_score(y_test, ypredabc))
        print("Recall Score:", recall_score(y_test, ypredabc))
        print("Precision Score:", precision_score(y_test, ypredabc))
        print("ROC AUC Score: ", roc_auc_score(y_test, ypredabc))
        abc_fp, abc_tp, abc_threshold = roc_curve(y_test, ypredabc)
        print("Threshold:", abc_threshold)

class AnonStandardModel:
    def __init__(self,args):
        self.kfile = str(args)+'.csv'
        self.csv_file="adult_anonymized_"+self.kfile

        
    def anonstandardmodel(self):
        csv_data=os.path.join("./results/adult/topdown",self.csv_file)
        df_data= pd.read_csv(csv_data, delimiter=';')
        df_data["native-country"].value_counts().index[0]
        col_names = df_data.columns
        df_data.drop('ID',axis=1)
        for c in col_names: 
            df_data = df_data.replace("?", np.NaN) 
            df_data = df_data.apply(lambda x:x.fillna(x.value_counts().index[0]))
            
        category_col =['age','workclass', 'education', 'marital-status', 'occupation', 
               'race', 'sex', 'native-country', 'salary-class']  
        labelEncoder = preprocessing.LabelEncoder() 
  
        mapping_dict ={} 
        for col in category_col: 
            df_data[col] = labelEncoder.fit_transform(df_data[col]) 
  
            le_name_mapping = dict(zip(labelEncoder.classes_, 
                        labelEncoder.transform(labelEncoder.classes_))) 
  
            mapping_dict[col]= le_name_mapping 
        Y = df_data['salary-class']
        X = df_data.drop('salary-class', axis = 1) 
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 100)
        tsne = TSNE(n_components=2, random_state=0)
        X_tsne = tsne.fit_transform(X.values)
        blue_patch = mpatches.Patch(color='#0A0AFF', label='<=50K')
        red_patch = mpatches.Patch(color='#AF0000', label='>50K')
        plt.figure(figsize=(20,10))
        plt.scatter(X_tsne[:,0], X_tsne[:,1], c=(Y == 0), cmap='coolwarm', label='<=50K', linewidths=2)
        plt.scatter(X_tsne[:,0], X_tsne[:,1], c=(Y == 1), cmap='coolwarm', label='>50K', linewidths=2)
        plt.title('t-SNE')
        plt.legend(handles=[blue_patch, red_patch])
        
        rf = RandomForestClassifier(n_estimators=100, random_state=0)
        rf.fit(X_train, y_train)
        ypredrf = rf.predict(X_test)
        print(confusion_matrix(y_test, ypredrf))
        print(classification_report(y_test, ypredrf))
        print("Accuracy Score:", accuracy_score(y_test, ypredrf))
        print("Recall Score:", recall_score(y_test, ypredrf))
        print("Precision Score:", precision_score(y_test, ypredrf))
        print("ROC AUC Score: ", roc_auc_score(y_test, ypredrf))
        rf_fp, rf_tp, rf_threshold = roc_curve(y_test, ypredrf)
        print("Threshold:", rf_threshold)
        
        gbc = GradientBoostingClassifier(n_estimators=100, random_state=0)
        gbc.fit(X_train, y_train)
        ypredgbc = gbc.predict(X_test)
        print(confusion_matrix(y_test, ypredgbc))
        print(classification_report(y_test, ypredgbc))
        print("Accuracy Score:", accuracy_score(y_test, ypredgbc))
        print("Recall Score:", recall_score(y_test, ypredgbc))
        print("Precision Score:", precision_score(y_test, ypredgbc))
        print("ROC AUC Score: ", roc_auc_score(y_test, ypredgbc))
        gbc_fp, gbc_tp, gbc_threshold = roc_curve(y_test, ypredgbc)
        print("Threshold:", gbc_threshold)
        
        abc = AdaBoostClassifier(n_estimators=100, random_state=0)
        abc.fit(X_train, y_train)
        ypredabc = abc.predict(X_test)
        print(confusion_matrix(y_test, ypredabc))
        print(classification_report(y_test, ypredabc))
        print("Accuracy Score:", accuracy_score(y_test, ypredabc))
        print("Recall Score:", recall_score(y_test, ypredabc))
        print("Precision Score:", precision_score(y_test, ypredabc))
        print("ROC AUC Score: ", roc_auc_score(y_test, ypredabc))
        abc_fp, abc_tp, abc_threshold = roc_curve(y_test, ypredabc)
        print("Threshold:", abc_threshold)

klist=[]

def main():
    for i in range (5):
        k = int(random.uniform(10, 2500))
        klist.append(k)
        args=["topdown",k, "adult"]
        
        anonymizer = Anonymizer(args)
        anonymizer.anonymize()
    standardmodel=StandardModel()
    standardmodel.standardmodel()
    
    
    for i in range(len(klist)):
        anonstandardmodel=AnonStandardModel(klist[i])
        anonstandardmodel.anonstandardmodel()
    
if __name__ == "__main__":
    main()


# args=str(1000)

# k_file = args+'.csv'
# csv_file="adult_anonymized_"+k_file

# print(csv_file)

# # Bounded region of parameter space
# pbounds = {'x': (10, 3000)}


# optimizer = BayesianOptimization(
#     f=anonymize,
#     pbounds=pbounds,
#     verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
#     random_state=1,
# )

