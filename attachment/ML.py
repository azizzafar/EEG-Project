#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, GroupShuffleSplit
from sklearn.decomposition import PCA
import matplotlib.patches as mpatches
import seaborn as sn
import warnings
warnings.filterwarnings('ignore')


from Hjorths import hjorths, getFeaturesIntoDataframe

class ML:
    def __init__(self, df):
        """
        This class is responsible for splitting the dataset into train and test sets.
        It performs the training of the three selected classifiers using the training
        dataset and evaluates their performance on the test dataset. The class returns
        various results, including accuracy, confusion matrix, and classification report.
        
        param:
            df: dataframe
        """
        self.df = df
        X = self.df.drop(["label", 'group_label'], axis=1)
        self.X = np.array(X)

        self.labels = self.df.label
        self.groups = self.df.group_label
        
        

        ## using grouKFold to make sure that data from a subject group 
        ## do not fall into both training set and test set.
        self.gkf = GroupKFold(n_splits = 5) 
        
        self.X_train_list = [] # list of folds for training
        self.y_train_list = [] # list of target/label for training

        self.X_test_list = [] # list of folds for validation
        self.y_test_list = [] # list of class labels for validation

        self.groups_train = [] # list of patients index in training data
        self.groups_test = [] # list of patiens index in validation data

        ## splitting data using groupKFold/GroupShuffleSplit
        for train_index, test_index in self.gkf.split(self.X, self.labels, groups = self.groups):
            X_train, y_train, g_train = self.X[train_index], self.labels[train_index], self.groups[train_index]
            X_test,  y_test, g_test  = self.X[test_index],  self.labels[test_index], self.groups[test_index]

            # scaling training and sience to get the zero mean
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.fit_transform(X_test)


            self.X_train_list.append(X_train)
            self.y_train_list.append(y_train)

            self.X_test_list.append(X_test)
            self.y_test_list.append(y_test)

            self.groups_train.append(g_train)
            self.groups_test.append(g_test)
            
        
        ####### 
        # load the classifiers results for easy access
        self.rf  = self.rfEstimator()
        self.gb  = self.gbEstimator()
        self.knn = self.knnEstimator()
        
    ############################################################################################
    def rfEstimator(self):
        
        """
        This function utilizes the train and test datasets that have been split using the
        GroupKFold technique. It performs cross-validation by using each fold as the test
        set once, while the remaining folds are used as the training set. The random forest
        (RF) classifier is applied to the training set, and the accuracy, confusion matrix, and 
        classification report are calculated using the validation set.
        
        return: 
            accuracy: dataframe, pandas dataframe
            cm:       list, list of pandas dataframe
            folds_test: dataframe, pandas dataframe
            folds_train: dataframe, pandas dataframe
            clr: list, list of pandas dataframe
        """
        
        df_train = []
        df_   = []
        cm = []
        accuracy = []
        clr = []
        for X_train,  y_train, X_test, y_test, g_train, g_test in zip(self.X_train_list, self.y_train_list, 
                                                        self.X_test_list, self.y_test_list, 
                                                        self.groups_train, self.groups_test):
            # Instantiate the machine learning classifier
            estimator = RandomForestClassifier(criterion= 'gini',
                                               max_features = 'auto',
                                               max_depth = 30,
                                               min_samples_leaf =1,
                                               min_samples_split = 6,
                                               n_estimators = 200,
                                               random_state = 21)

            # Fit the classifier to the training data
            estimator.fit(X_train, y_train)

            # Predict the classes of the testing data
            y_pred = estimator.predict(X_test)

            #cm = confusion_matrix(y_test, y_pred)

            # Use pandas for computation of confusion matrix
            # We want to save the labels 

            accuracy.append(accuracy_score(y_test, y_pred))
            df = pd.DataFrame({'y_true': y_test, 
                               'y_pred': y_pred})
            cm.append(pd.crosstab(df.y_true, df.y_pred))

            clr.append(pd.DataFrame(classification_report(y_test, y_pred, output_dict = True, zero_division = 0)))
            
            df['groups'] = g_test
            df_.append(df)
            
            col = ['groups']
            
            dd = pd.DataFrame({'class' : y_train, 
                               'groups': g_train})
            df_train.append(dd)
            
         
        ## Patient in list of test-folds
        patients_idx = []
        fold_idx = []
        for i in range(0, len(df_)):
            patients_idx.append(df_[i].groups.unique())
            fold_idx.append(f'fold {i}')
        Folds = pd.DataFrame(patients_idx, index = fold_idx)
        Folds.columns = [''] * len(Folds.columns)
        
        
        ## groups in training dataset
        patients_idx2 = [] # list of patients in training set. 
        fold_idx2 = [] # list of folds 
        for i in range(0, len(df_train)):
            patients_idx2.append(df_train[i].groups.unique()) # find patients in fold i (0,,,len(fold)) 
            fold_idx2.append(f'fold {i}')
        Folds2 = pd.DataFrame(patients_idx2, index = fold_idx2) # convert list of folds into DataFrame.
        Folds2.columns = [''] * len(Folds2.columns) # 
        
        ## Accuracy results for each folds 
        col = ['Accuracy'] # list of accuracy for folds
        accuracy = pd.DataFrame(accuracy, index = fold_idx, columns = col) # convert list of accurcay into DataFrame.
        
        return {'accuracy': accuracy, 'cm': cm, 'folds_test': Folds,'folds_train': Folds2 ,'clr': clr}

    def gbEstimator(self):
        
        """
        This function utilizes the train and test datasets that have been split using the
        GroupKFold technique. It performs cross-validation by using each fold as the test
        set once, while the remaining folds are used as the training set. The gradient boosting
        (GB) classifier is applied to the training set, and the accuracy, confusion matrix, and 
        classification report are calculated using the validation set.
        
        return: 
            accuracy: dataframe, pandas dataframe
            cm:       list, list of pandas dataframe
            folds_test: dataframe, pandas dataframe
            folds_train: dataframe, pandas dataframe
            clr: list, list of pandas dataframe
        """
        
        df_train = []
        df_   = []
        cm = []
        accuracy = []
        clr = []
        for X_train,  y_train, X_test, y_test, g_train, g_test in zip(self.X_train_list, self.y_train_list, 
                                                            self.X_test_list, self.y_test_list, 
                                                            self.groups_train, self.groups_test):
            # Instantiate the machine learning classifier
            #estimator = GradientBoostingClassifier(loss = self.gs_rg.best_params_.get('loss'),
             #                                      n_estimators=self.gs_rg.best_params_.get('n_estimators'),
              #                                     min_samples_leaf =  self.gs_rg.best_params_.get('min_samples_leaf'),
               #                                    learning_rate= self.gs_rg.best_params_.get('learning_rate'),
                #                                   max_depth=self.gs_rg.best_params_.get('max_depth'),
                 #                                  random_state=21)
            estimator = GradientBoostingClassifier(loss = 'deviance',
                                                   n_estimators=100,
                                                   min_samples_leaf = 0.1,
                                                   learning_rate= 0.05,
                                                   max_depth=8,
                                                   random_state=21)
     
            # Fit the classifier to the training data
            estimator.fit(X_train, y_train)

            # Predict the classes of the testing data
            y_pred = estimator.predict(X_test)

            #cm = confusion_matrix(y_test, y_pred)

            # Use pandas for computation of confusion matrix
            # We want to save the labels 

            accuracy.append(accuracy_score(y_test, y_pred))
            df = pd.DataFrame({'y_true': y_test, 
                               'y_pred': y_pred})
            cm.append(pd.crosstab(df.y_true, df.y_pred))

            clr.append(pd.DataFrame(classification_report(y_test, y_pred, output_dict = True, zero_division = 0)))
            
            df['groups'] = g_test
            df_.append(df)
            
            col = ['groups']
            
            dd = pd.DataFrame({'class' : y_train, 
                               'groups': g_train})
            df_train.append(dd)
            
         
        ## Patient in list of test-folds
        patients_idx = []
        fold_idx = []
        for i in range(0, len(df_)):
            patients_idx.append(df_[i].groups.unique())
            fold_idx.append(f'fold {i}')
        Folds = pd.DataFrame(patients_idx, index = fold_idx)
        Folds.columns = [''] * len(Folds.columns)
        
        
        ## groups in training dataset
        patients_idx2 = [] # list of patients in training set. 
        fold_idx2 = [] # list of folds 
        for i in range(0, len(df_train)):
            patients_idx2.append(df_train[i].groups.unique()) # find patients in fold i (0,,,len(fold)) 
            fold_idx2.append(f'fold {i}')
        Folds2 = pd.DataFrame(patients_idx2, index = fold_idx2) # convert list of folds into DataFrame.
        Folds2.columns = [''] * len(Folds2.columns) # 
        
        ## Accuracy results for each folds 
        col = ['Accuracy'] # list of accuracy for folds
        accuracy = pd.DataFrame(accuracy, index = fold_idx, columns = col)#convert list of accurcay into DataFrame.
        
        return {'accuracy': accuracy, 'cm': cm, 'folds_test': Folds,'folds_train': Folds2 ,'clr': clr}


    
    def knnEstimator(self):
        """
        This function utilizes the train and test datasets that have been split using the
        GroupKFold technique. It performs cross-validation by using each fold as the test
        set once, while the remaining folds are used as the training set. The K-Nearest Neighbor
        (KNN) classifier is applied to the training set, and the accuracy, confusion matrix, and 
        classification report are calculated using the validation set.
        
        return: 
            accuracy: dataframe, pandas dataframe
            cm:       list, list of pandas dataframe
            folds_test: dataframe, pandas dataframe
            folds_train: dataframe, pandas dataframe
            clr: list, list of pandas dataframe
        """
        df_train = []
        df_   = []
        cm = []
        accuracy = []
        clr = []
        for X_train,  y_train, X_test, y_test, g_train, g_test in zip(self.X_train_list, self.y_train_list, 
                                                      self.X_test_list, self.y_test_list,
                                                      self.groups_train, self.groups_test):
        
            # Instantiate the machine learning classifier
            estimator= KNeighborsClassifier(metric = 'manhattan',
                                            n_neighbors= 3,
                                           weights = 'distance')

            # Fit the classifier to the training data
            estimator.fit(X_train, y_train)

            # Predict the classes of the testing data
            y_pred = estimator.predict(X_test)

            #cm = confusion_matrix(y_test, y_pred)

            # Use pandas for computation of confusion matrix
            # We want to save the labels 

            accuracy.append(accuracy_score(y_test, y_pred))
            df = pd.DataFrame({'y_true': y_test, 
                               'y_pred': y_pred})
            cm.append(pd.crosstab(df.y_true, df.y_pred))
            
            # Compute classification report for each fold and store it
            clr.append(pd.DataFrame(classification_report(y_test, y_pred, 
                                                          output_dict = True, zero_division = 0)))
            # group of the patient in the test set
            df['groups'] = g_test
            df_.append(df)
            
            col = ['groups']
            
            dd = pd.DataFrame({'class' : y_train, 
                               'groups': g_train})
            df_train.append(dd)
            
         
        ## list of subjects in test-folds
        patients_idx = []
        fold_idx = []
        for i in range(0, len(df_)):
            patients_idx.append(df_[i].groups.unique())
            fold_idx.append(f'fold {i}')
        Folds = pd.DataFrame(patients_idx, index = fold_idx)
        Folds.columns = [''] * len(Folds.columns)
        
        
        # list of subjects in training set
        patients_idx2 = [] # list of patients in training set. 
        fold_idx2 = [] # list of folds 
        for i in range(0, len(df_train)):
            patients_idx2.append(df_train[i].groups.unique()) # find patients in fold i (0,,,len(fold)) 
            fold_idx2.append(f'fold {i}')
        Folds2 = pd.DataFrame(patients_idx2, index = fold_idx2) # convert list of folds into DataFrame.
        Folds2.columns = [''] * len(Folds2.columns) # 
        
        ## Accuracy results for each folds 
        col = ['Accuracy'] # list of accuracy for folds
        accuracy = pd.DataFrame(accuracy, index = fold_idx, columns = col)#convert list of accurcay into DataFrame.
        
        return {'accuracy': accuracy, 'cm': cm, 'folds_test': Folds,'folds_train': Folds2 ,'clr': clr}


# In[4]:


class get_ML_Results:
    def __init__(self, DLB_Standard, AD_Standard, PDD_Standard, PD_Standard, HC_Standard,
                 DLB_Target,AD_Target, PDD_Target, PD_Target, HC_Target,
                DLB_Distractor,AD_Distractor, PDD_Distractor, PD_Distractor, HC_Distractor):
        
        """
        In this class, the "getFeaturesIntoDataframe" class is used to organize the final
        datasets for the classification task. This class takes the necessary input data
        and processes it to create a structured dataframe that is suitable for classification.
        Once the datasets are prepared, the ML class is utilized to perform the classification
        of the subjects. The ML class implements machine learning algorithms and techniques to
        train and evaluate models based on the provided datasets.
        
        
        params:
            DLB_Standard: object, object of the class that is used to claculate the Hjorth's 
                          parameters for DLB subject group.
            AD_Standard: object, object of the class that is used to claculate the Hjorth's 
                         parameters for AD subject group.                      
            PDD_Standard: object, object of the class that is used to claculate the Hjorth's 
                          parameters for PDD subject group. 
            PD_Standard: object, object of the class that is used to claculate the Hjorth's
                         parameters for PD subject group. 
            HC_Standard: object, object of the class that is used to claculate the Hjorth's
                         parameters for HC subject group. 
            
            DLB_Target: object, object of the class that is used to claculate the Hjorth's 
                        parameters for DLB subject group. 
            AD_Target:  object, object of the class that is used to claculate the Hjorth's 
                        parameters for AD subject group. 
            PDD_Target: object, object of the class that is used to claculate the Hjorth's
                        parameters for PDD subject group. 
            PD_Target:  object, object of the class that is used to claculate the Hjorth's
                        parameters for PD subject group. 
            HC_Target:  object, object of the class that is used to claculate the Hjorth's
                        parameters for HC subject group.
            
            DLB_Distractor: object, object of the class that is used to claculate the Hjorth's
                            parameters for DLB subject group.
            AD_Distractor: object, object of the class that is used to claculate the Hjorth's
                            parameters for DLB subject group. 
            PDD_Distracor: object, object of the class that is used to claculate the Hjorth's
                            parameters for DLB subject group.
            PD_Distractor: object, object of the class that is used to claculate the Hjorth's
                            parameters for DLB subject group. 
            HC_Distractor: object, object of the class that is used to claculate the Hjorth's
                            parameters for DLB subject group. 
            
        """
        
        self.DLB_Standard = DLB_Standard
        self.AD_Standard  = AD_Standard
        self.PDD_Standard = PDD_Standard
        self.PD_Standard  = PD_Standard
        self.HC_Standard  = HC_Standard
        
        self.DLB_Target = DLB_Target
        self.AD_Target  = AD_Target
        self.PDD_Target = PDD_Target
        self.PD_Target  = PD_Target
        self.HC_Target  = HC_Target
        
        self.DLB_Distractor = DLB_Distractor
        self.AD_Distractor  = AD_Distractor
        self.PDD_Distractor = PDD_Distractor
        self.PD_Distractor  = PD_Distractor
        self.HC_Distractor  = HC_Distractor
        self.num_channels = 9
        
        ################################################################
        ## Activity parameter of Hjorth's parameters
        # Standard epochs
        dfAS = getFeaturesIntoDataframe(DLB = self.DLB_Standard, AD = self.AD_Standard, 
                                                       PDD = self.PDD_Standard, PD = self.PD_Standard,
                                                       HC = self.HC_Standard, num_channels = self.num_channels, Activity = True,
                                                       Mobility = False, Complexity =False,
                                                       bundle = False)
        self.dfAS   = dfAS.getDataset()
        # Target epochs 
        dfAT = getFeaturesIntoDataframe(DLB = self.DLB_Target, AD = self.AD_Target, 
                                               PDD = self.PDD_Target, PD = self.PD_Target,
                                               HC = self.HC_Target, num_channels = self.num_channels, Activity = True,
                                               Mobility = False, Complexity =False,
                                               bundle = False)
        self.dfAT      = dfAT.getDataset()
        
        # Distractor epochs
        dfAD = getFeaturesIntoDataframe(DLB = self.DLB_Distractor, AD = self.AD_Distractor, 
                                               PDD = self.PDD_Distractor, PD = self.PD_Distractor,
                                               HC = self.HC_Distractor, num_channels = self.num_channels, Activity = True,
                                               Mobility = False, Complexity =False,
                                               bundle = False)
        self.dfAD      = dfAD.getDataset()
        ################################################################
        ## Mobility parameter of Hjorth's parameters
        # Standard epochs
        dfMS = getFeaturesIntoDataframe(DLB = self.DLB_Standard, AD = self.AD_Standard, 
                                                       PDD = self.PDD_Standard, PD = self.PD_Standard,
                                                       HC = self.HC_Standard, num_channels = self.num_channels, Activity = False,
                                                       Mobility = True, Complexity =False,
                                                       bundle = False)
        self.dfMS   = dfMS.getDataset()
        
        # Target epochs
        dfMT = getFeaturesIntoDataframe(DLB = self.DLB_Target, AD = self.AD_Target, 
                                               PDD = self.PDD_Target, PD = self.PD_Target,
                                               HC = self.HC_Target, num_channels = self.num_channels, Activity = False,
                                               Mobility = True, Complexity =False,
                                               bundle = False)
        self.dfMT      = dfMT.getDataset()
        
        # Distractor epochs 
        dfMD = getFeaturesIntoDataframe(DLB = self.DLB_Distractor, AD = self.AD_Distractor, 
                                               PDD = self.PDD_Distractor, PD = self.PD_Distractor,
                                               HC = self.HC_Distractor, num_channels = self.num_channels, Activity = False,
                                               Mobility = True, Complexity =False,
                                               bundle = False)
        self.dfMD      = dfMD.getDataset()
        ################################################################
        ## complexity feature of Hjorth's parameters
        # Standard epochs 
        dfCS = getFeaturesIntoDataframe(DLB = self.DLB_Standard, AD = self.AD_Standard, 
                                                       PDD = self.PDD_Standard, PD = self.PD_Standard,
                                                       HC = self.HC_Standard, num_channels = self.num_channels, Activity = False,
                                                       Mobility = False, Complexity =True,
                                                       bundle = False)
        self.dfCS   = dfCS.getDataset()
       
        # Target epochs
        dfCT = getFeaturesIntoDataframe(DLB = self.DLB_Target, AD = self.AD_Target, 
                                               PDD = self.PDD_Target, PD = self.PD_Target,
                                               HC = self.HC_Target, num_channels = self.num_channels, Activity = False,
                                               Mobility = False, Complexity =True,
                                               bundle = False)
        self.dfCT      = dfCT.getDataset()
        
        # Distractor epochs
        dfCD = getFeaturesIntoDataframe(DLB = self.DLB_Distractor, AD = self.AD_Distractor, 
                                               PDD = self.PDD_Distractor, PD = self.PD_Distractor,
                                               HC = self.HC_Distractor, num_channels = self.num_channels, Activity = False,
                                               Mobility = False, Complexity =True,
                                               bundle = False)
        self.dfCD      = dfCD.getDataset()
        
        
        self.mlAS   = ML(self.dfAS)#Machine-Learning classification applied on activity features of standard epochs
        self.mlAT   = ML(self.dfAT)#Machine-Learning classification applied on activity features of target epochs
        self.mlAD   = ML(self.dfAD)# 
        
        self.mlMS   = ML(self.dfMS)#Machine-Learning classification applied on mobility features of standard epochs
        self.mlMT   = ML(self.dfMT)#Machine-Learning classification applied on mobility features of target epochs
        self.mlMD   = ML(self.dfMD)# 
        
        self.mlCS   = ML(self.dfCS)#Machine-Learning classification applied on complexity features of standard epochs
        self.mlCT   = ML(self.dfCT)#Machine-Learning classification applied on complexity features of target epochs
        self.mlCD   = ML(self.dfCD)# 


# In[5]:


class highlight_max_clr:
    def __init__(self, Results ,Standard = False, Target = False, Distractor = False):
        
        """
        In this class, the results of evaluation metrics such as f1-score, precision,
        and recall,  obtained by the "get_ML_Results" class, are utilized to determine
        the best-performing classifier among the RF, GB, and KNN classifiers. The class
        starts by organizing these results into a table format, allowing for easy 
        comparison and analysis. The metrics scores of each classifier are then compared
        to identify the classifier with the highest performance. The best-performing 
        classifier that exhibits the most accurate classification results based on the
        specified metrics are then highlighted with a color. 
        
        
        parameters: 
            Results: class, object, (The running "get_ML_Results" class)
            
            Standard: False|True
                When the parameter is set to True, this class will utilize the results
                obtained from ML classifiers for the features of standard stimuli, namely
                activity, mobility, and complexity. It will compare these results and identify
                the best performer among the classifiers. This comparison and determination
                of the best performer will be executed when the function "mean_std" is called.
                
            Target: False|True
                When the parameter is set to True, this class will utilize the results
                obtained from ML classifiers for the features of target stimuli, namely
                activity, mobility, and complexity. It will compare these results and identify
                the best performer among the classifiers. This comparison and determination
                of the best performer will be executed when the function "mean_std" is called.
                
            Distractor: False|True
                When the parameter is set to True, this class will utilize the results
                obtained from ML classifiers for the features of distractor stimuli, namely
                activity, mobility, and complexity. It will compare these results and identify
                the best performer among the classifiers. This comparison and determination
                of the best performer will be executed when the function "mean_std" is called.
                
        """
        
        
        self.Results = Results
        self.Standard = Standard
        self.Target = Target
        self.Distractor = Distractor
        
        # concatenate the classification report from each of the datasets and classifires. 
        # use only f1-score, precision and recall part.
        if self.Standard == True:
            self.meanRFA = pd.concat(self.Results.mlAS.rf.get('clr')).groupby(level=0).mean()[:3] # Activity
            self.meanRFM = pd.concat(self.Results.mlMS.rf.get('clr')).groupby(level=0).mean()[:3] # Mobility
            self.meanRFC = pd.concat(self.Results.mlCS.rf.get('clr')).groupby(level=0).mean()[:3] # Complexity

            # Confuison matrix representing Gradient Boosting results for Standard epochs 
            self.meanGBA = pd.concat(self.Results.mlAS.gb.get('clr')).groupby(level=0).mean()[:3] # Activity
            self.meanGBM = pd.concat(self.Results.mlMS.gb.get('clr')).groupby(level=0).mean()[:3] # Mobility
            self.meanGBC = pd.concat(self.Results.mlCS.gb.get('clr')).groupby(level=0).mean()[:3] # Complexity


            # Confuison matrix representing Random kNN results for Standard epochs 
            self.meanKNNA = pd.concat(self.Results.mlAS.knn.get('clr')).groupby(level=0).mean()[:3] # Activity
            self.meanKNNM = pd.concat(self.Results.mlMS.knn.get('clr')).groupby(level=0).mean()[:3] # Mobility
            self.meanKNNC = pd.concat(self.Results.mlCS.knn.get('clr')).groupby(level=0).mean()[:3] # Complexity
            
            self.stdRFA = pd.concat(self.Results.mlAS.rf.get('clr')).groupby(level=0).std()[:3] # Activity
            self.stdRFM = pd.concat(self.Results.mlMS.rf.get('clr')).groupby(level=0).std()[:3] # Mobility
            self.stdRFC = pd.concat(self.Results.mlCS.rf.get('clr')).groupby(level=0).std()[:3] # Complexity

            # Confuison matrix representing Gradient Boosting results for Standard epochs 
            self.stdGBA = pd.concat(self.Results.mlAS.gb.get('clr')).groupby(level=0).std()[:3] # Activity
            self.stdGBM = pd.concat(self.Results.mlMS.gb.get('clr')).groupby(level=0).std()[:3] # Mobility
            self.stdGBC = pd.concat(self.Results.mlCS.gb.get('clr')).groupby(level=0).std()[:3] # Complexity


            # Confuison matrix representing Random kNN results for Standard epochs 
            self.stdKNNA = pd.concat(self.Results.mlAS.knn.get('clr')).groupby(level=0).std()[:3] # Activity
            self.stdKNNM = pd.concat(self.Results.mlMS.knn.get('clr')).groupby(level=0).std()[:3] # Mobility
            self.stdKNNC = pd.concat(self.Results.mlCS.knn.get('clr')).groupby(level=0).std()[:3] # Complexity
            
            
            
        elif self.Target == True:
            self.meanRFA = pd.concat(self.Results.mlAT.rf.get('clr')).groupby(level=0).mean()[:3] # Activity
            self.meanRFM = pd.concat(self.Results.mlMT.rf.get('clr')).groupby(level=0).mean()[:3] # Mobility
            self.meanRFC = pd.concat(self.Results.mlCT.rf.get('clr')).groupby(level=0).mean()[:3] # Complexity

            # Confuison matrix representing Gradient Boosting results for Standard epochs 
            self.meanGBA = pd.concat(self.Results.mlAT.gb.get('clr')).groupby(level=0).mean()[:3] # Activity
            self.meanGBM = pd.concat(self.Results.mlMT.gb.get('clr')).groupby(level=0).mean()[:3] # Mobility
            self.meanGBC = pd.concat(self.Results.mlCT.gb.get('clr')).groupby(level=0).mean()[:3] # Complexity


            # Confuison matrix representing Random kNN results for Standard epochs 
            self.meanKNNA = pd.concat(self.Results.mlAT.knn.get('clr')).groupby(level=0).mean()[:3] # Activity
            self.meanKNNM = pd.concat(self.Results.mlMT.knn.get('clr')).groupby(level=0).mean()[:3] # Mobility
            self.meanKNNC = pd.concat(self.Results.mlCT.knn.get('clr')).groupby(level=0).mean()[:3] # Complexity
            
            
            self.stdRFA = pd.concat(self.Results.mlAT.rf.get('clr')).groupby(level=0).std()[:3] # Activity
            self.stdRFM = pd.concat(self.Results.mlMT.rf.get('clr')).groupby(level=0).std()[:3] # Mobility
            self.stdRFC = pd.concat(self.Results.mlCT.rf.get('clr')).groupby(level=0).std()[:3] # Complexity

            # Confuison matrix representing Gradient Boosting results for Standard epochs 
            self.stdGBA = pd.concat(self.Results.mlAT.gb.get('clr')).groupby(level=0).std()[:3] # Activity
            self.stdGBM = pd.concat(self.Results.mlMT.gb.get('clr')).groupby(level=0).std()[:3] # Mobility
            self.stdGBC = pd.concat(self.Results.mlCT.gb.get('clr')).groupby(level=0).std()[:3] # Complexity


            # Confuison matrix representing Random kNN results for Standard epochs 
            self.stdKNNA = pd.concat(self.Results.mlAT.knn.get('clr')).groupby(level=0).std()[:3] # Activity
            self.stdKNNM = pd.concat(self.Results.mlMT.knn.get('clr')).groupby(level=0).std()[:3] # Mobility
            self.stdKNNC = pd.concat(self.Results.mlCT.knn.get('clr')).groupby(level=0).std()[:3] # Complexity
            
            
        elif self.Distractor == True:
            self.meanRFA = pd.concat(self.Results.mlAD.rf.get('clr')).groupby(level=0).mean()[:3] # Activity
            self.meanRFM = pd.concat(self.Results.mlMD.rf.get('clr')).groupby(level=0).mean()[:3] # Mobility
            self.meanRFC = pd.concat(self.Results.mlCD.rf.get('clr')).groupby(level=0).mean()[:3] # Complexity

            # Confuison matrix representing Gradient Boosting results for Standard epochs 
            self.meanGBA = pd.concat(self.Results.mlAD.gb.get('clr')).groupby(level=0).mean()[:3] # Activity
            self.meanGBM = pd.concat(self.Results.mlMD.gb.get('clr')).groupby(level=0).mean()[:3] # Mobility
            self.meanGBC = pd.concat(self.Results.mlCD.gb.get('clr')).groupby(level=0).mean()[:3] # Complexity


            # Confuison matrix representing Random kNN results for Standard epochs 
            self.meanKNNA = pd.concat(self.Results.mlAD.knn.get('clr')).groupby(level=0).mean()[:3] # Activity
            self.meanKNNM = pd.concat(self.Results.mlMD.knn.get('clr')).groupby(level=0).mean()[:3] # Mobility
            self.meanKNNC = pd.concat(self.Results.mlCD.knn.get('clr')).groupby(level=0).mean()[:3] # Complexity
            
            
            self.stdRFA = pd.concat(self.Results.mlAD.rf.get('clr')).groupby(level=0).std()[:3] # Activity
            self.stdRFM = pd.concat(self.Results.mlMD.rf.get('clr')).groupby(level=0).std()[:3] # Mobility
            self.stdRFC = pd.concat(self.Results.mlCD.rf.get('clr')).groupby(level=0).std()[:3] # Complexity

            # Confuison matrix representing Gradient Boosting results for Standard epochs 
            self.stdGBA = pd.concat(self.Results.mlAD.gb.get('clr')).groupby(level=0).std()[:3] # Activity
            self.stdGBM = pd.concat(self.Results.mlMD.gb.get('clr')).groupby(level=0).std()[:3] # Mobility
            self.stdGBC = pd.concat(self.Results.mlCD.gb.get('clr')).groupby(level=0).std()[:3] # Complexity


            # Confuison matrix representing Random kNN results for Standard epochs 
            self.stdKNNA = pd.concat(self.Results.mlAD.knn.get('clr')).groupby(level=0).std()[:3] # Activity
            self.stdKNNM = pd.concat(self.Results.mlMD.knn.get('clr')).groupby(level=0).std()[:3] # Mobility
            self.stdKNNC = pd.concat(self.Results.mlCD.knn.get('clr')).groupby(level=0).std()[:3] # Complexity
        else:
            pass
        
        
        #############################################3
        # load the results for the "clr_mean", and "clr_std" for easy access 
        self.df_meanA = self.clr_mean(input_feature = 'Activity')
        self.df_stdA = self.clr_std(input_feature= 'Activity')
        # find the numeric parts of dataframe, without column and indexes. 
        data_A = []
        for val in self.df_meanA.values:
            val_list = []
            for j in val:
                val_list.append(j)
            data_A.append(val_list[:5])
        df_A = pd.DataFrame(data_A)
        # find position of max values 
        self.pos_idx_A = self.find_pos_val_max(df_A)
        
        ###################################################
        # load the results for the "clr_mean", and "clr_std" for easy access 
        self.df_meanM= self.clr_mean(input_feature = 'Mobility')
        self.df_stdM = self.clr_std(input_feature= 'Mobility')
        # find the numeric parts of dataframe, without column and indexes. 
        data_M = []
        for val in self.df_meanM.values:
            val_list = []
            for j in val:
                val_list.append(j)
            data_M.append(val_list[:5])
        df_M = pd.DataFrame(data_M)
        # find position of max values 
        self.pos_idx_M = self.find_pos_val_max(df_M)
        
        ########################################################
        # load the results for the "clr_mean", and "clr_std" for easy access 
        self.df_meanC = self.clr_mean(input_feature = 'Complexity')
        self.df_stdC = self.clr_std(input_feature= 'Complexity')
        # find the numeric parts of dataframe, without column and indexes. 
        data_C = []
        for val in self.df_meanC.values:
            val_list = []
            for j in val:
                val_list.append(j)
            data_C.append(val_list[:5])
        df_C = pd.DataFrame(data_C)
        # find position of max values 
        self.pos_idx_C = self.find_pos_val_max(df_C)
    

    def clr_mean(self, input_feature = None):
        
        """
        This function is designed to generate a table or dataframe that facilitates
        the comparison of the average metrics, such as f1-score, precision, and recall.
        It takes the average of each metric, calculated in the "__init__," and organizes
        them into a table or dataframe format, allowing for easy visual comparison 
        and analysis of the performance of the classifiers based on these metrics.
        
        parameter: 
            input_feature: string, name of the feature such as "Activity", "Mobility" or 
            "Complexity"
        
        return:
            df: dataframe, containing the mean of metrics for the five-fold cross-validation.
        """
        
        if self.Standard == True:
            if input_feature == 'Activity':
                df = pd.concat([self.meanRFA, self.meanGBA, self.meanKNNA])

                input_features = ['Activity','Activity','Activity', 'Activity','Activity','Activity', 
                                  'Activity','Activity','Activity']
                approachs = ['Random Forest', 'Random Forest', 'Random Forest', 'Gradient Boosting','Gradient Boosting',
                             'Gradient Boosting','KNN', 'KNN', 'KNN']
                dataset = ['Standard','Standard','Standard','Standard','Standard','Standard',
                          'Standard','Standard','Standard']


                df = df.T[:5]
                df = df.T
                df['Input Features'] = input_features
                df['Approach'] = approachs
                df['Dataset'] = dataset


            elif input_feature == 'Mobility':
                df = pd.concat([self.meanRFM, self.meanGBM, self.meanKNNM])

                input_features = ['Mobility','Mobility','Mobility', 'Mobility','Mobility','Mobility', 
                                  'Mobility','Mobility','Mobility']
                approachs = ['Random Forest', 'Random Forest', 'Random Forest', 'Gradient Boosting','Gradient Boosting',
                             'Gradient Boosting','KNN', 'KNN', 'KNN']
                dataset = ['Standard','Standard','Standard','Standard','Standard','Standard',
                          'Standard','Standard','Standard']


                df = df.T[:5]
                df = df.T
                df['Input Features'] = input_features
                df['Approach'] = approachs
                df['Dataset'] = dataset


            elif input_feature == 'Complexity':
                df = pd.concat([self.meanRFC, self.meanGBC, self.meanKNNC])

                input_features = ['Complexity','Complexity','Complexity', 'Complexity','Complexity','Complexity', 
                                  'Complexity','Complexity','Complexity']
                approachs = ['Random Forest', 'Random Forest', 'Random Forest', 'Gradient Boosting','Gradient Boosting',
                             'Gradient Boosting','KNN', 'KNN', 'KNN']
                dataset = ['Standard','Standard','Standard','Standard','Standard','Standard',
                          'Standard','Standard','Standard']


                df = df.T[:5]
                df = df.T
                df['Input Features'] = input_features
                df['Approach'] = approachs
                df['Dataset'] = dataset

            else:
                raise ValueError("Choose input_feature e.g. input_feature = 'Activity', 'Mobility', or 'Complexity'")
        
        elif self.Target == True:
            if input_feature == 'Activity':
                df = pd.concat([self.meanRFA, self.meanGBA, self.meanKNNA])

                input_features = ['Activity','Activity','Activity', 'Activity','Activity','Activity', 
                                  'Activity','Activity','Activity']
                approachs = ['Random Forest', 'Random Forest', 'Random Forest', 'Gradient Boosting','Gradient Boosting',
                             'Gradient Boosting','KNN', 'KNN', 'KNN']
                dataset = ['Target','Target','Target','Target','Target','Target',
                          'Target','Target','Target']


                df = df.T[:5]
                df = df.T
                df['Input Features'] = input_features
                df['Approach'] = approachs
                df['Dataset'] = dataset


            elif input_feature == 'Mobility':
                df = pd.concat([self.meanRFM, self.meanGBM, self.meanKNNM])

                input_features = ['Mobility','Mobility','Mobility', 'Mobility','Mobility','Mobility', 
                                  'Mobility','Mobility','Mobility']
                approachs = ['Random Forest', 'Random Forest', 'Random Forest', 'Gradient Boosting','Gradient Boosting',
                             'Gradient Boosting','KNN', 'KNN', 'KNN']
                dataset = ['Target','Target','Target','Target','Target','Target',
                          'Target','Target','Target']


                df = df.T[:5]
                df = df.T
                df['Input Features'] = input_features
                df['Approach'] = approachs
                df['Dataset'] = dataset


            elif input_feature == 'Complexity':
                df = pd.concat([self.meanRFC, self.meanGBC, self.meanKNNC])

                input_features = ['Complexity','Complexity','Complexity', 'Complexity','Complexity','Complexity', 
                                  'Complexity','Complexity','Complexity']
                approachs = ['Random Forest', 'Random Forest', 'Random Forest', 'Gradient Boosting','Gradient Boosting',
                             'Gradient Boosting','KNN', 'KNN', 'KNN']
                dataset = ['Target','Target','Target','Target','Target','Target',
                          'Target','Target','Target']


                df = df.T[:5]
                df = df.T
                df['Input Features'] = input_features
                df['Approach'] = approachs
                df['Dataset'] = dataset

            else:
                raise ValueError("Choose input_feature e.g. input_feature = 'Activity', 'Mobility', or 'Complexity'")
        
        elif self.Distractor == True:
            if input_feature == 'Activity':
                df = pd.concat([self.meanRFA, self.meanGBA, self.meanKNNA])

                input_features = ['Activity','Activity','Activity', 'Activity','Activity','Activity', 
                                  'Activity','Activity','Activity']
                approachs = ['Random Forest', 'Random Forest', 'Random Forest', 'Gradient Boosting','Gradient Boosting',
                             'Gradient Boosting','KNN', 'KNN', 'KNN']
                dataset = ['Distractor','Distractor','Distractor','Distractor','Distractor','Distractor',
                          'Distractor','Distractor','Distractor']


                df = df.T[:5]
                df = df.T
                df['Input Features'] = input_features
                df['Approach'] = approachs
                df['Dataset'] = dataset


            elif input_feature == 'Mobility':
                df = pd.concat([self.meanRFM, self.meanGBM, self.meanKNNM])

                input_features = ['Mobility','Mobility','Mobility', 'Mobility','Mobility','Mobility', 
                                  'Mobility','Mobility','Mobility']
                approachs = ['Random Forest', 'Random Forest', 'Random Forest', 'Gradient Boosting','Gradient Boosting',
                             'Gradient Boosting','KNN', 'KNN', 'KNN']
                dataset = ['Distractor','Distractor','Distractor','Distractor','Distractor','Distractor',
                          'Distractor','Distractor','Distractor']


                df = df.T[:5]
                df = df.T
                df['Input Features'] = input_features
                df['Approach'] = approachs
                df['Dataset'] = dataset


            elif input_feature == 'Complexity':
                df = pd.concat([self.meanRFC, self.meanGBC, self.meanKNNC])

                input_features = ['Complexity','Complexity','Complexity', 'Complexity','Complexity','Complexity', 
                                  'Complexity','Complexity','Complexity']
                approachs = ['Random Forest', 'Random Forest', 'Random Forest', 'Gradient Boosting','Gradient Boosting',
                             'Gradient Boosting','KNN', 'KNN', 'KNN']
                dataset = ['Distractor','Distractor','Distractor','Distractor','Distractor','Distractor',
                          'Distractor','Distractor','Distractor']


                df = df.T[:5]
                df = df.T
                df['Input Features'] = input_features
                df['Approach'] = approachs
                df['Dataset'] = dataset

            else:
                raise ValueError("Choose input_feature e.g. input_feature = 'Activity', 'Mobility', or 'Complexity'")
        else:
            pass
            
        
            

        return df

    
    ################# Standard deviation of precision, recall and f1-score for the folds ###############
    def clr_std(self, input_feature = None):
        """
        This function is designed to generate a table or dataframe that facilitates
        the comparison of the standard deveiation for metrics, such as f1-score, precision,
        and recall. It takes the standard deviation of each metric, calculated in the "__init__," and
        organizes them into a table or dataframe format, allowing for easy visual comparison 
        and analysis of the performance of the classifiers based on these metrics.
        
        parameter: 
            input_feature: string, name of the feature such as "Activity", "Mobility" or 
            "Complexity"
            
        return: 
            df: dataframe, df containing the standard deviation for the five-fold cross-validation. 
        
        """
        
        if self.Standard == True:
            if input_feature == 'Activity':
                df = pd.concat([self.stdRFA, self.stdGBA, self.stdKNNA])

                input_features = ['Activity','Activity','Activity', 'Activity','Activity','Activity', 
                                  'Activity','Activity','Activity']
                approachs = ['Random Forest', 'Random Forest', 'Random Forest', 'Gradient Boosting','Gradient Boosting',
                             'Gradient Boosting','KNN', 'KNN', 'KNN']
                dataset = ['Standard','Standard','Standard','Standard','Standard','Standard',
                          'Standard','Standard','Standard']


                df = df.T[:5]
                df = df.T
                df['Input Features'] = input_features
                df['Approach'] = approachs
                df['Dataset'] = dataset


            elif input_feature == 'Mobility':
                df = pd.concat([self.stdRFM, self.stdGBM, self.stdKNNM])

                input_features = ['Mobility','Mobility','Mobility', 'Mobility','Mobility','Mobility', 
                                  'Mobility','Mobility','Mobility']
                approachs = ['Random Forest', 'Random Forest', 'Random Forest', 'Gradient Boosting','Gradient Boosting',
                             'Gradient Boosting','KNN', 'KNN', 'KNN']
                dataset = ['Standard','Standard','Standard','Standard','Standard','Standard',
                          'Standard','Standard','Standard']


                df = df.T[:5]
                df = df.T
                df['Input Features'] = input_features
                df['Approach'] = approachs
                df['Dataset'] = dataset


            elif input_feature == 'Complexity':
                df = pd.concat([self.stdRFC, self.stdGBC, self.stdKNNC])

                input_features = ['Complexity','Complexity','Complexity', 'Complexity','Complexity','Complexity', 
                                  'Complexity','Complexity','Complexity']
                approachs = ['Random Forest', 'Random Forest', 'Random Forest', 'Gradient Boosting','Gradient Boosting',
                             'Gradient Boosting','KNN', 'KNN', 'KNN']
                dataset = ['Standard','Standard','Standard','Standard','Standard','Standard',
                          'Standard','Standard','Standard']


                df = df.T[:5]
                df = df.T
                df['Input Features'] = input_features
                df['Approach'] = approachs
                df['Dataset'] = dataset

            else:
                raise ValueError("Choose input_feature e.g. input_feature = 'Activity', 'Mobility', or 'Complexity'")
                
                ###############################################################
        elif self.Target == True:
            if input_feature == 'Activity':
                df = pd.concat([self.stdRFA, self.stdGBA, self.stdKNNA])

                input_features = ['Activity','Activity','Activity', 'Activity','Activity','Activity', 
                                  'Activity','Activity','Activity']
                approachs = ['Random Forest', 'Random Forest', 'Random Forest', 'Gradient Boosting','Gradient Boosting',
                             'Gradient Boosting','KNN', 'KNN', 'KNN']
                dataset = ['Target','Target','Target','Target','Target','Target',
                          'Target','Target','Target']


                df = df.T[:5]
                df = df.T
                df['Input Features'] = input_features
                df['Approach'] = approachs
                df['Dataset'] = dataset


            elif input_feature == 'Mobility':
                df = pd.concat([self.stdRFM, self.stdGBM, self.stdKNNM])

                input_features = ['Mobility','Mobility','Mobility', 'Mobility','Mobility','Mobility', 
                                  'Mobility','Mobility','Mobility']
                approachs = ['Random Forest', 'Random Forest', 'Random Forest', 'Gradient Boosting','Gradient Boosting',
                             'Gradient Boosting','KNN', 'KNN', 'KNN']
                dataset = ['Target','Target','Target','Target','Target','Target',
                          'Target','Target','Target']


                df = df.T[:5]
                df = df.T
                df['Input Features'] = input_features
                df['Approach'] = approachs
                df['Dataset'] = dataset

                
            elif input_feature == 'Complexity':
                df = pd.concat([self.stdRFC, self.stdGBC, self.stdKNNC])

                input_features = ['Complexity','Complexity','Complexity', 'Complexity','Complexity','Complexity', 
                                  'Complexity','Complexity','Complexity']
                approachs = ['Random Forest', 'Random Forest', 'Random Forest', 'Gradient Boosting','Gradient Boosting',
                             'Gradient Boosting','KNN', 'KNN', 'KNN']
                dataset = ['Target','Target','Target','Target','Target','Target',
                          'Target','Target','Target']


                df = df.T[:5]
                df = df.T
                df['Input Features'] = input_features
                df['Approach'] = approachs
                df['Dataset'] = dataset

            else:
                raise ValueError("Choose input_feature e.g. input_feature = 'Activity', 'Mobility', or 'Complexity'")
                
                #################################################################
        elif self.Distractor == True:
            if input_feature == 'Activity':
                df = pd.concat([self.stdRFA, self.stdGBA, self.stdKNNA])

                input_features = ['Activity','Activity','Activity', 'Activity','Activity','Activity', 
                                  'Activity','Activity','Activity']
                approachs = ['Random Forest', 'Random Forest', 'Random Forest', 'Gradient Boosting','Gradient Boosting',
                             'Gradient Boosting','KNN', 'KNN', 'KNN']
                dataset = ['Distractor','Distractor','Distractor','Distractor','Distractor','Distractor',
                          'Distractor','Distractor','Distractor']


                df = df.T[:5]
                df = df.T
                df['Input Features'] = input_features
                df['Approach'] = approachs
                df['Dataset'] = dataset


            elif input_feature == 'Mobility':
                df = pd.concat([self.stdRFM, self.stdGBM, self.stdKNNM])

                input_features = ['Mobility','Mobility','Mobility', 'Mobility','Mobility','Mobility', 
                                  'Mobility','Mobility','Mobility']
                approachs = ['Random Forest', 'Random Forest', 'Random Forest', 'Gradient Boosting','Gradient Boosting',
                             'Gradient Boosting','KNN', 'KNN', 'KNN']
                dataset = ['Distractor','Distractor','Distractor','Distractor','Distractor','Distractor',
                          'Distractor','Distractor','Distractor']


                df = df.T[:5]
                df = df.T
                df['Input Features'] = input_features
                df['Approach'] = approachs
                df['Dataset'] = dataset

                
            elif input_feature == 'Complexity':
                df = pd.concat([self.stdRFC, self.stdGBC, self.stdKNNC])

                input_features = ['Complexity','Complexity','Complexity', 'Complexity','Complexity','Complexity', 
                                  'Complexity','Complexity','Complexity']
                approachs = ['Random Forest', 'Random Forest', 'Random Forest', 'Gradient Boosting','Gradient Boosting',
                             'Gradient Boosting','KNN', 'KNN', 'KNN']
                dataset = ['Distractor','Distractor','Distractor','Distractor','Distractor','Distractor',
                          'Distractor','Distractor','Distractor']


                df = df.T[:5]
                df = df.T
                df['Input Features'] = input_features
                df['Approach'] = approachs
                df['Dataset'] = dataset

            else:
                raise ValueError("Choose input_feature e.g. input_feature = 'Activity', 'Mobility', or 'Complexity'")




        return df
    
    def find_pos_val_max(self, df):
        """
        This function is utilized to identify the maximum values and their corresponding
        positions for the metrics of f1-score, precision, and recall.
        
        parameter: 
            df: dataframe of only numeric values
            
        return: 
            idx_max_f1:  int, number (index number of the f1-score with the max value) 
            pos_max_f1:  int, number (column number of the f1-score with the max value)
            idx_max_prc: int, number (index number of the precision with the max value)  
            pos_max_prc: int, number (column number of the precision with the max value)
            idx_max_rec: int, number (index number of the recall with the max value) 
            pos_max_rec: int, number (column number of the recall with the max value)  
        """
        positions = df.idxmax(axis = 1)
        values = df.max(axis=1)


        val_f1  = [values[0], values[3], values[6]]
        val_prc = [values[1], values[4], values[7]]
        val_rec = [values[2], values[5], values[8]]

        pos_f1  = [positions[0], positions[3], positions[6]]
        pos_prc = [positions[1], positions[4], positions[7]]
        pos_rec = [positions[2], positions[5], positions[8]]

        idx_f1 = [0, 3, 6]
        idx_prc = [1, 4, 7]
        idx_rec = [2, 5, 8]


        idx_max_f1 = 0
        pos_max_f1 = 0
        val_max_f1 = 0
        for i, j, k in zip(val_f1,pos_f1 ,idx_f1):
            if i >= val_max_f1:
                val_max_f1 = i
                pos_max_f1 = j
                idx_max_f1 = k

        idx_max_prc = 0
        pos_max_prc = 0
        val_max_prc = 0
        for i, j, k in zip(val_prc,pos_prc ,idx_prc):
            if i >= val_max_prc:
                val_max_prc = i
                pos_max_prc = j
                idx_max_prc = k

        idx_max_rec = 0
        pos_max_rec = 0
        val_max_rec = 0
        for i, j, k in zip(val_rec, pos_rec ,idx_rec):
            if i >= val_max_rec:
                val_max_rec = i
                pos_max_rec = j
                idx_max_rec = k

        return idx_max_f1, pos_max_f1, idx_max_prc, pos_max_prc, idx_max_rec, pos_max_rec




    def style_specific_cell_Activity(self, A):
        """
        param:
            A: dataframe, df for activity feature
        
        return:
            df: dataframe, highlighted with max values for the metrics.
        """
        color1 = 'background-color: lightgreen'
        color2 = 'background-color: red'
        color3 = 'background-color: yellow'

        df = pd.DataFrame('', index=A.index, columns=A.columns)
        df.iloc[self.pos_idx_A[0], self.pos_idx_A[1]] = color1
        df.iloc[self.pos_idx_A[2], self.pos_idx_A[3]] = color2
        df.iloc[self.pos_idx_A[4], self.pos_idx_A[5]] = color3
        return df
    
    def style_specific_cell_Mobility(self, M):
        """
        param:
            M: dataframe, df for mobility feature
        
        return:
            df: dataframe, highlighted with max values for the metrics.
        """
        #color = label(len(pos), 'background-color: lightgreen')
        color1 = 'background-color: lightgreen'
        color2 = 'background-color: red'
        color3 = 'background-color: yellow'

        df = pd.DataFrame('', index=M.index, columns=M.columns)
        df.iloc[self.pos_idx_M[0], self.pos_idx_M[1]] = color1
        df.iloc[self.pos_idx_M[2], self.pos_idx_M[3]] = color2
        df.iloc[self.pos_idx_M[4], self.pos_idx_M[5]] = color3
        return df

    def style_specific_cell_Complexity(self, C):
        """
        param:
            C: dataframe, df for complexity feature
         
        return:
            df: dataframe, highlighted with max values for the metrics.
        """
        #color = label(len(pos), 'background-color: lightgreen')
        color1 = 'background-color: lightgreen'
        color2 = 'background-color: red'
        color3 = 'background-color: yellow'

        df = pd.DataFrame('', index=C.index, columns=C.columns)
        df.iloc[self.pos_idx_C[0], self.pos_idx_C[1]] = color1
        df.iloc[self.pos_idx_C[2], self.pos_idx_C[3]] = color2
        df.iloc[self.pos_idx_C[4], self.pos_idx_C[5]] = color3
        return df
    
    def mean_std(self, input_feature = None):
        
        """
        In this function, the "clr_mean," "clr_std," and "style_specific_cells" functions
        are combined to generate the final table that includes the average values of 
        the metrics, their corresponding standard deviations, and highlights the best
        performer. The "clr_mean" function calculates the mean values for each metric,
        the "clr_std" function computes the standard deviations, and the "style_specific_cells"
        function highlights the cells with the highest metric values.
        
        param:
            input_feature: string, the name of the feature or dataset.
            
        return: dataframe, highlighted dataframe with the best performer.
        """ 
        if input_feature == 'Activity':
            dfmean =self.df_meanA
            dfstd = self.df_stdA
            df_mean_std = pd.DataFrame()
            df_mean_std['AD'] = dfmean['AD'].round(decimals = 3).astype('str') + ' ( ' + dfstd['AD'].round(decimals = 2).astype('str') + ')'
            df_mean_std['DLB'] = dfmean['DLB'].round(decimals = 3).astype('str') + ' ( ' + dfstd['DLB'].round(decimals = 2).astype('str') + ')'
            df_mean_std['HC'] = dfmean['HC'].round(decimals = 3).astype('str') + ' ( ' + dfstd['HC'].round(decimals = 2).astype('str') + ')'
            df_mean_std['PD'] = dfmean['PD'].round(decimals = 3).astype('str') + ' ( ' + dfstd['PD'].round(decimals = 2).astype('str') + ')'
            df_mean_std['PDD'] = dfmean['PDD'].round(decimals = 3).astype('str') + ' ( ' + dfstd['PDD'].round(decimals = 2).astype('str') + ')'
            df_mean_std['Input Feature'] = dfmean['Input Features']
            df_mean_std['Dataset'] = dfmean['Dataset']
            df_mean_std['Method'] = ['RF', 'RF', 'RF', 'GB', 'GB', 'GB', 'KNN', 'KNN', 'KNN']

            index = ['f1-score/Random Forest', 'precision/Random Forest', 'recall/Random Forest', 
                     'f1-score/Gradient Boosting','precision/Gradient Boosting', 'recall/Gradient Boosting',
                      'f1-score/KNN', 'precision/KNN', 'recall/KNN']

            df_mean_std.index = index
            df_mean_std = df_mean_std.drop([ 'Input Feature','Dataset'], axis = 1)

            df_final = df_mean_std.style.apply(self.style_specific_cell_Activity, axis = None)

            return df_final

        elif input_feature == 'Mobility':
            dfmean =self.df_meanM
            dfstd = self.df_stdM

            df_mean_std = pd.DataFrame()
            df_mean_std['AD'] = dfmean['AD'].round(decimals = 3).astype('str') + ' ( ' + dfstd['AD'].round(decimals = 2).astype('str') + ')'
            df_mean_std['DLB'] = dfmean['DLB'].round(decimals = 3).astype('str') + ' ( ' + dfstd['DLB'].round(decimals = 2).astype('str') + ')'
            df_mean_std['HC'] = dfmean['HC'].round(decimals = 3).astype('str') + ' ( ' + dfstd['HC'].round(decimals = 2).astype('str') + ')'
            df_mean_std['PD'] = dfmean['PD'].round(decimals = 3).astype('str') + ' ( ' + dfstd['PD'].round(decimals = 2).astype('str') + ')'
            df_mean_std['PDD'] = dfmean['PDD'].round(decimals = 3).astype('str') + ' ( ' + dfstd['PDD'].round(decimals = 2).astype('str') + ')'
            df_mean_std['Input Feature'] = dfmean['Input Features']
            df_mean_std['Method'] = ['RF', 'RF', 'RF', 'GB', 'GB', 'GB', 'KNN', 'KNN', 'KNN']

            index = ['f1-score/Random Forest', 'precision/Random Forest', 'recall/Random Forest', 
                     'f1-score/Gradient Boosting','precision/Gradient Boosting', 'recall/Gradient Boosting',
                      'f1-score/KNN', 'precision/KNN', 'recall/KNN']
            df_mean_std['Dataset'] = dfmean['Dataset']

            df_mean_std.index = index
            df_mean_std = df_mean_std.drop(['Input Feature','Dataset'], axis = 1)

            df_final = df_mean_std.style.apply(self.style_specific_cell_Mobility, axis = None)

            return df_final


        elif input_feature == 'Complexity':
            dfmean =self.df_meanC
            dfstd = self.df_stdC

            df_mean_std = pd.DataFrame()
           
            df_mean_std['AD'] = dfmean['AD'].round(decimals = 3).astype('str') + ' ( ' + dfstd['AD'].round(decimals = 2).astype('str') + ')'
            df_mean_std['DLB'] = dfmean['DLB'].round(decimals = 3).astype('str') + ' ( ' + dfstd['DLB'].round(decimals = 2).astype('str') + ')'
            df_mean_std['HC'] = dfmean['HC'].round(decimals = 3).astype('str') + ' ( ' + dfstd['HC'].round(decimals = 2).astype('str') + ')'
            df_mean_std['PD'] = dfmean['PD'].round(decimals = 3).astype('str') + ' ( ' + dfstd['PD'].round(decimals = 2).astype('str') + ')'
            df_mean_std['PDD'] = dfmean['PDD'].round(decimals = 3).astype('str') + ' ( ' + dfstd['PDD'].round(decimals = 2).astype('str') + ')'
            df_mean_std['Input Feature'] = dfmean['Input Features']
            #df_mean_std['Approach'] = dfmean['Approach']
            df_mean_std['Dataset'] = dfmean['Dataset']
            df_mean_std['Method'] = ['RF', 'RF', 'RF', 'GB', 'GB', 'GB', 'KNN', 'KNN', 'KNN']
          
           
            index = ['f1-score/Random Forest', 'precision/Random Forest', 'recall/Random Forest', 
                     'f1-score/Gradient Boosting','precision/Gradient Boosting', 'recall/Gradient Boosting',
                      'f1-score/KNN', 'precision/KNN', 'recall/KNN']

            df_mean_std.index = index
            df_mean_std = df_mean_std.drop(['Input Feature', 'Dataset'], axis = 1)

            df_final = df_mean_std.style.apply(self.style_specific_cell_Complexity, axis = None)

            return df_final
        else:
            raise ValueError("Choose input_feature e.g. input_feature = 'Activity', 'Mobility', or 'Complexity'")
            


# In[6]:


class highlight_max_acc:
    def __init__(self, Results):
        
        """
        In this class, the accuracy results obtained from the "get_ML_Results" class
        are utilized to determine the best performing classifier among the RF, GB, 
        and KNN classifiers. The class starts by organizing these results into a table
        format, allowing for easy comparison and analysis. The accuracy scores of each
        classifier are then examined and compared to identify the classifier with the
        highest accuracy. This best performer is subsequently highlighted, indicating
        the classifier that demonstrates the most accurate classification results 
        based on the accuracy metric.
        
        parameter:
                Results: class object,  results of "get_ML_Results" class . 
        """
        
        self.Results = Results
    
        #############################################
        self.df_mean_S   =  self.mean_acc( dataset = 'Standard')
        self.df_mean_T   =  self.mean_acc( dataset = 'Target')
        self.df_mean_D   =  self.mean_acc( dataset = 'Distractor')
       
        self.df_std_S    =  self.std_acc( dataset = 'Standard')
        self.df_std_T    =  self.std_acc( dataset = 'Target')
        self.df_std_D    =  self.std_acc( dataset = 'Distractor')
        
        self.idx_max_S = self.max_pos_index_acc(self.df_mean_S)
        self.idx_max_T = self.max_pos_index_acc(self.df_mean_T)
        self.idx_max_D = self.max_pos_index_acc(self.df_mean_D)
        
    ###########################################################################3    
    def mean_acc(self, dataset = None):
        
        """
        This function is designed to generate a table or dataframe that facilitates the
        comparison of average accuracies. It takes the accuracies obtained from different
        classifiers, calculates their averages for the five-fold cross-validation, and arranges
        them in a tabular format. 
        
        param: 
            dataset: str, name of the stimuli such as (Standard, Target or Distractor)
        
        return: DataFrame
        
        """
        if dataset == 'Standard':                 
            df  = pd.DataFrame([
                ['Activity', 'Random Forest','Standard',round(self.Results.mlAS.rf.get('accuracy').Accuracy.mean(),2)],
                ['Activity', 'Gradient Boosting','Standard',round(self.Results.mlAS.gb.get('accuracy').Accuracy.mean(),2)],
                ['Activity', 'K-Nearest Neighbors','Standard',round(self.Results.mlAS.knn.get('accuracy').Accuracy.mean(),2)],
                ['Mobility', 'Random Forest','Standard',round(self.Results.mlMS.rf.get('accuracy').Accuracy.mean(),2)],
                ['Mobility', 'Gradient Boosting','Standard',round(self.Results.mlMS.gb.get('accuracy').Accuracy.mean(),2)],
                ['Mobility', 'K-Nearest Neighbors','Standard',round(self.Results.mlMS.knn.get('accuracy').Accuracy.mean(),2)],
                ['Complexity', 'Random Forest','Standard',round(self.Results.mlCS.rf.get('accuracy').Accuracy.mean(),2)],
                ['Complexity', 'Gradient Boosting','Standard',round(self.Results.mlCS.gb.get('accuracy').Accuracy.mean(),2)],
                ['Complexity', 'K-Nearest Neighbors','Standard',round(self.Results.mlCS.knn.get('accuracy').Accuracy.mean(),2)]
                ])
        elif dataset == 'Target':
            df = pd.DataFrame([
                ['Activity', 'Random Forest','Target',round(self.Results.mlAT.rf.get('accuracy').Accuracy.mean(),2)],
                ['Activity', 'Gradient Boosting','Target',round(self.Results.mlAT.gb.get('accuracy').Accuracy.mean(),2)],
                ['Activity', 'K-Nearest Neighbors','Target',round(self.Results.mlAT.knn.get('accuracy').Accuracy.mean(),2)],
                ['Mobility', 'Random Forest','Target',round(self.Results.mlMT.rf.get('accuracy').Accuracy.mean(),2)],
                ['Mobility', 'Gradient Boosting','Target',round(self.Results.mlMT.gb.get('accuracy').Accuracy.mean(),2)],
                ['Mobility', 'K-Nearest Neighbors','Target',round(self.Results.mlMT.knn.get('accuracy').Accuracy.mean(),2)],
                ['Complexity', 'Random Forest','Target',round(self.Results.mlCT.rf.get('accuracy').Accuracy.mean(),2)],
                ['Complexity', 'Gradient Boosting','Target',round(self.Results.mlCT.gb.get('accuracy').Accuracy.mean(),2)],
                ['Complexity', 'K-Nearest Neighbors','Target',round(self.Results.mlCT.knn.get('accuracy').Accuracy.mean(),2)]
                ])
        elif dataset == 'Distractor':
            df = pd.DataFrame([
                ['Activity', 'Random Forest','Distractor',round(self.Results.mlAD.rf.get('accuracy').Accuracy.mean(),2)],
                ['Activity', 'Gradient Boosting','Distractor',round(self.Results.mlAD.gb.get('accuracy').Accuracy.mean(),2)],
                ['Activity', 'K-Nearest Neighbors','Distractor',round(self.Results.mlAD.knn.get('accuracy').Accuracy.mean(),2)],
                ['Mobility', 'Random Forest','Distractor',round(self.Results.mlMD.rf.get('accuracy').Accuracy.mean(),2)],
                ['Mobility', 'Gradient Boosting','Distractor',round(self.Results.mlMD.gb.get('accuracy').Accuracy.mean(),2)],
                ['Mobility', 'K-Nearest Neighbors','Distractor',round(self.Results.mlMD.knn.get('accuracy').Accuracy.mean(),2)],
                ['Complexity', 'Random Forest','Distractor',round(self.Results.mlCD.rf.get('accuracy').Accuracy.mean(),2)],
                ['Complexity', 'Gradient Boosting','Distractor',round(self.Results.mlCD.gb.get('accuracy').Accuracy.mean(),2)],
                ['Complexity', 'K-Nearest Neighbors','Distractor',round(self.Results.mlCD.knn.get('accuracy').Accuracy.mean(),2)]  
                ])
        else:
            raise ValueError("Choose a dataset e.g. dataset = 'Standard', 'Target', or 'Distractor'")
            
        col = ['Input Feature', 'Approach','Dataset', 'Accuracy']
        df.columns = col
        
        df = df.drop(['Dataset'], axis = 1)
        return df
    
    def std_acc(self, dataset = None):
        
        """
        This function is designed to generate a table or dataframe that facilitates the
        comparison of standard deviation for accuracies. It takes the accuracies obtained from different
        classifiers, calculates their standard deviation for the five-fold cross-validation, and arranges
        them in a tabular format. 
        
        param: 
            dataset: str, name of the stimuli such as (Standard, Target or Distractor)
        
        return: DataFrame
        
        """
        if dataset == 'Standard':                 
            df  = pd.DataFrame([
                ['Activity', 'Random Forest','Standard',round(self.Results.mlAS.rf.get('accuracy').Accuracy.std(),2)],
                ['Activity', 'Gradient Boosting','Standard',round(self.Results.mlAS.gb.get('accuracy').Accuracy.std(),2)],
                ['Activity', 'K-Nearest Neighbors','Standard',round(self.Results.mlAS.knn.get('accuracy').Accuracy.std(),2)],
                ['Mobility', 'Random Forest','Standard',round(self.Results.mlMS.rf.get('accuracy').Accuracy.std(),2)],
                ['Mobility', 'Gradient Boosting','Standard',round(self.Results.mlMS.gb.get('accuracy').Accuracy.std(),2)],
                ['Mobility', 'K-Nearest Neighbors','Standard',round(self.Results.mlMS.knn.get('accuracy').Accuracy.std(),2)],
                ['Complexity', 'Random Forest','Standard',round(self.Results.mlCS.rf.get('accuracy').Accuracy.std(),2)],
                ['Complexity', 'Gradient Boosting','Standard',round(self.Results.mlCS.gb.get('accuracy').Accuracy.std(),2)],
                ['Complexity', 'K-Nearest Neighbors','Standard',round(self.Results.mlCS.knn.get('accuracy').Accuracy.std(),2)]
                ])
        elif dataset == 'Target':
            df = pd.DataFrame([
                ['Activity', 'Random Forest','Target',round(self.Results.mlAT.rf.get('accuracy').Accuracy.std(),2)],
                ['Activity', 'Gradient Boosting','Target',round(self.Results.mlAT.gb.get('accuracy').Accuracy.std(),2)],
                ['Activity', 'K-Nearest Neighbors','Target',round(self.Results.mlAT.knn.get('accuracy').Accuracy.std(),2)],
                ['Mobility', 'Random Forest','Target',round(self.Results.mlMT.rf.get('accuracy').Accuracy.std(),2)],
                ['Mobility', 'Gradient Boosting','Target',round(self.Results.mlMT.gb.get('accuracy').Accuracy.std(),2)],
                ['Mobility', 'K-Nearest Neighbors','Target',round(self.Results.mlMT.knn.get('accuracy').Accuracy.std(),2)],
                ['Complexity', 'Random Forest','Target',round(self.Results.mlCT.rf.get('accuracy').Accuracy.std(),2)],
                ['Complexity', 'Gradient Boosting','Target',round(self.Results.mlCT.gb.get('accuracy').Accuracy.std(),2)],
                ['Complexity', 'K-Nearest Neighbors','Target',round(self.Results.mlCT.knn.get('accuracy').Accuracy.std(),2)]
                ])
        elif dataset == 'Distractor':
            df = pd.DataFrame([
                ['Activity', 'Random Forest','Distractor',round(self.Results.mlAD.rf.get('accuracy').Accuracy.std(),2)],
                ['Activity', 'Gradient Boosting','Distractor',round(self.Results.mlAD.gb.get('accuracy').Accuracy.std(),2)],
                ['Activity', 'K-Nearest Neighbors','Distractor',round(self.Results.mlAD.knn.get('accuracy').Accuracy.std(),2)],
                ['Mobility', 'Random Forest','Distractor',round(self.Results.mlMD.rf.get('accuracy').Accuracy.std(),2)],
                ['Mobility', 'Gradient Boosting','Distractor',round(self.Results.mlMD.gb.get('accuracy').Accuracy.std(),2)],
                ['Mobility', 'K-Nearest Neighbors','Distractor',round(self.Results.mlMD.knn.get('accuracy').Accuracy.std(),2)],
                ['Complexity', 'Random Forest','Distractor',round(self.Results.mlCD.rf.get('accuracy').Accuracy.std(),2)],
                ['Complexity', 'Gradient Boosting','Distractor',round(self.Results.mlCD.gb.get('accuracy').Accuracy.std(),2)],
                ['Complexity', 'K-Nearest Neighbors','Distractor',round(self.Results.mlCD.knn.get('accuracy').Accuracy.std(),2)]  
                ])
        else:
            raise ValueError("Choose a dataset e.g. dataset = 'Standard', 'Target', or 'Distractor'")
            
        col = ['Input Feature', 'Approach', 'Dataset', 'Accuracy']
        df.columns = col
        df = df.drop(['Dataset'], axis = 1)
        return df
    
    def max_pos_index_acc(self, df):
        val_acc = df.Accuracy
        val_index = np.arange(0, len(val_acc))
        idx_max_f1 = 0
        val_max_f1 = 0
        for i, j in zip(val_index, val_acc):
            if j >= val_max_f1:
                val_max_f1 = j
                idx_max_f1 = i
        return idx_max_f1

    def style_specific_cell_Standard(self, x):
        #color = label(len(pos), 'background-color: lightgreen')
        color1 = 'background-color: lightgreen'
        df = pd.DataFrame('', index=x.index, columns=x.columns)
        df.iloc[self.idx_max_S, 2] = color1
        return df
    def style_specific_cell_Target(self, x):
        #color = label(len(pos), 'background-color: lightgreen')
        color1 = 'background-color: lightgreen'
        df = pd.DataFrame('', index=x.index, columns=x.columns)
        df.iloc[self.idx_max_T, 2] = color1
        return df
    def style_specific_cell_Distractor(self, x):
        #color = label(len(pos), 'background-color: lightgreen')
        color1 = 'background-color: lightgreen'
        df = pd.DataFrame('', index=x.index, columns=x.columns)
        df.iloc[self.idx_max_D, 2] = color1
        return df
    
    def mean_std_accuracy(self, dataset = None):
        
        """
        In this function, the "mean_acc," "std_acc," and "style_specific_cells" functions
        are combined to generate the final table that includes the average values of the 
        accuracies, their corresponding standard deviations, and highlights the best performer. 
        
        param:
            dataset: str, name of the stimuli such as ("Standard", "Target", or "Distractor")
           
        return:
            df: dataframe, highlighted with max value.
        """
        if dataset == 'Standard':
            dfmean = self.df_mean_S
            dfstd = self.df_std_S
            df_mean_std = pd.DataFrame()
            df_mean_std['Input Feature'] = dfmean['Input Feature']
            df_mean_std['Approach'] = dfmean['Approach']
            #df_mean_std['Dataset'] = dfmean['Dataset']         
            df_mean_std['Accuracy'] = (dfmean['Accuracy'].round(decimals = 3).astype('str') +
                                       ' ( ' + dfstd['Accuracy'].round(decimals = 2).astype('str') + ')')

            df = df_mean_std.style.apply(self.style_specific_cell_Standard, axis = None)
            return df
        
        elif dataset == 'Target':
            dfmean = self.df_mean_T
            dfstd = self.df_std_T
            df_mean_std = pd.DataFrame()
            df_mean_std['Input Feature'] = dfmean['Input Feature']
            df_mean_std['Approach'] = dfmean['Approach']
            #df_mean_std['Dataset'] = dfmean['Dataset']         
            df_mean_std['Accuracy'] = (dfmean['Accuracy'].round(decimals = 3).astype('str') +
                                       ' ( ' + dfstd['Accuracy'].round(decimals = 2).astype('str') + ')')

            df = df_mean_std.style.apply(self.style_specific_cell_Target, axis = None)
            return df
        
        elif dataset == 'Distractor':
            dfmean = self.df_mean_D
            dfstd = self.df_std_D
            df_mean_std = pd.DataFrame()
            df_mean_std['Input Feature'] = dfmean['Input Feature']
            df_mean_std['Approach'] = dfmean['Approach']
            #df_mean_std['Dataset'] = dfmean['Dataset']         
            df_mean_std['Accuracy'] = (dfmean['Accuracy'].round(decimals = 3).astype('str') +
                                       ' ( ' + dfstd['Accuracy'].round(decimals = 2).astype('str') + ')')

            df = df_mean_std.style.apply(self.style_specific_cell_Distractor, axis = None)
            return df
        else:
            raise ValueError("Choose a dataset e.g. dataset = 'Standard', 'Target', or 'Distractor'")
              
        
        
class plotCM:
    def __init__(self, Results):
        
        """
        This class is used to visualize the confusion matrix for the features 
        of standard stimuli, target stimuli, and distractor stimuli. It utilizes
        the results obtained from the classifiers, including Random Forest,
        Gradient Boosting, and K-Nearest Neighbor classifier.
        
        parameters: 
            Results: class , (The running "get_ML_Results" class)
        """
        
        self.Results = Results        
    def plot_CM_standard(self):
        
        """
        This function is used to plot the confusion matrices for the features of standard stimuli. 
        It begins by averaging the list of confusion matrices obtained for each fold. By averaging
        these matrices, we obtain a consolidated representation of the overall performance of the
        classifiers on the standard stimuli features. The resulting averaged confusion matrix is 
        then visualized using the seaborn's heatmap. 
        """
        
        # Confuison matrix representing Random Forest results  
        cmRFAS = pd.concat(self.Results.mlAS.rf.get('cm')).groupby(level=0).mean().round(0)  # Activity
        cmRFMS = pd.concat(self.Results.mlMS.rf.get('cm')).groupby(level=0).mean().round(0)  # Mobility
        cmRFCS = pd.concat(self.Results.mlCS.rf.get('cm')).groupby(level=0).mean().round(0)  # Complexity
        
        # Confuison matrix representing Gradient Boosting results
        cmGBAS = pd.concat(self.Results.mlAS.gb.get('cm')).groupby(level=0).mean().round(0)  # Activity
        cmGBMS = pd.concat(self.Results.mlMS.gb.get('cm')).groupby(level=0).mean().round(0)  # Mobility
        cmGBCS = pd.concat(self.Results.mlCS.gb.get('cm')).groupby(level=0).mean().round(0)  # Complexity
        
        # Confuison matrix representing Random kNN results
        cmKNNAS = pd.concat(self.Results.mlAS.knn.get('cm')).groupby(level=0).mean().round(0)  # Activity
        cmKNNMS = pd.concat(self.Results.mlMS.knn.get('cm')).groupby(level=0).mean().round(0)  # Mobility
        cmKNNCS = pd.concat(self.Results.mlCS.knn.get('cm')).groupby(level=0).mean().round(0)  # Complexity

        
        # Visualization of confusion matrixes for activity parameters
        plt.figure(figsize=(13,4),  tight_layout=True)
        plt.subplot(1, 3, 1)
        sn.heatmap(cmRFAS,annot=True, cmap ='Spectral') # Random Forest / Activity
        plt.title('Random Forest')

        plt.subplot(1, 3, 2)
        sn.heatmap(cmGBAS,annot=True, cmap ='Spectral') # Gradient Boosting / Activity
        plt.title('Gradient Boosting')

        plt.subplot(1, 3, 3)
        sn.heatmap(cmKNNAS,annot=True, cmap ='Spectral')# kNN / Activity
        plt.title('K-Nearest Neighbors')
        plt.suptitle('Activity feature of Standard Stimuli')

        #Visualization of confusion matrixes for Mobility parameters
        plt.figure(figsize=(13,4),  tight_layout=True)
        plt.subplot(1, 3, 1)
        sn.heatmap(cmRFMS,annot=True, cmap ='Spectral') # RF / Mobility
        plt.title('Random Forest')

        plt.subplot(1, 3, 2)
        sn.heatmap(cmGBMS,annot=True, cmap ='Spectral') # GB / Mobility
        plt.title('Gradient Boosting')

        plt.subplot(1, 3, 3)
        sn.heatmap(cmKNNMS,annot=True, cmap ='Spectral') # kNN / Mobility
        plt.title('K-Nearest Neighbors')
        plt.suptitle('Mobility feature of Standard Stimuli')
        
        
        ##Visualization of confusion matrixes for Complexity parameters
        plt.figure(figsize=(13,4),  tight_layout=True)
        plt.subplot(1, 3, 1)
        sn.heatmap(cmRFCS,annot=True, cmap ='Spectral')# Random Forest / Complexity
        plt.title('Random Forest')

        plt.subplot(1, 3, 2)
        sn.heatmap(cmGBCS,annot=True, cmap ='Spectral') # GB / Complexity
        plt.title('Gradient Boosting')

        plt.subplot(1, 3, 3)
        sn.heatmap(cmKNNCS,annot=True, cmap ='Spectral') # kNN / Complexity
        plt.title('K-Nearest Neighbors')
        plt.suptitle('Complexity feature of Standard Stimuli')
        plt.show()
    
    def plot_CM_target(self):
        
        """
        This function is used to plot the confusion matrices for the features of target stimuli. 
        It begins by averaging the list of confusion matrices obtained for each fold. By averaging
        these matrices, we obtain a consolidated representation of the overall performance of the
        classifiers on the standard stimuli features. The resulting averaged confusion matrix is 
        then visualized using the seaborn's heatmap. 
        """
        
        # Confuison matrix representing Random Forest results  
        cmRFAT = pd.concat(self.Results.mlAT.rf.get('cm')).groupby(level=0).mean().round(0)  # Activity
        cmRFMT = pd.concat(self.Results.mlMT.rf.get('cm')).groupby(level=0).mean().round(0) # Mobility
        cmRFCT = pd.concat(self.Results.mlCT.rf.get('cm')).groupby(level=0).mean().round(0) # Complexity
        
        # Confuison matrix representing Gradient Boosting results
        cmGBAT = pd.concat(self.Results.mlAT.gb.get('cm')).groupby(level=0).mean().round(0)  # Activity
        cmGBMT = pd.concat(self.Results.mlMT.gb.get('cm')).groupby(level=0).mean().round(0)  # Mobility
        cmGBCT = pd.concat(self.Results.mlCT.gb.get('cm')).groupby(level=0).mean().round(0)  # Complexity
        
        # Confuison matrix representing kNN results
        cmKNNAT = pd.concat(self.Results.mlAT.knn.get('cm')).groupby(level=0).mean().round(0)  # Activity
        cmKNNMT = pd.concat(self.Results.mlMT.knn.get('cm')).groupby(level=0).mean().round(0)  # Mobility
        cmKNNCT = pd.concat(self.Results.mlCT.knn.get('cm')).groupby(level=0).mean().round(0)  # Complexity

        
        # Visualization of confusion matrixes for activity parameters
        plt.figure(figsize=(13,4),  tight_layout=True)
        plt.subplot(1, 3, 1)
        sn.heatmap(cmRFAT,annot=True, cmap ='Spectral') # Random Forest / Activity
        plt.title('Random Forest')

        plt.subplot(1, 3, 2)
        sn.heatmap(cmGBAT,annot=True, cmap ='Spectral') # Gradient Boosting / Activity
        plt.title('Gradient Boosting')

        plt.subplot(1, 3, 3)
        sn.heatmap(cmKNNAT,annot=True, cmap ='Spectral')# kNN / Activity
        plt.title('K-Nearest Neighbors')
        plt.suptitle('Activity feature of Target Stimuli')

        #Visualization of confusion matrixes for Mobility parameters
        plt.figure(figsize=(13,4),  tight_layout=True)
        plt.subplot(1, 3, 1)
        sn.heatmap(cmRFMT,annot=True, cmap ='Spectral') # RF / Mobility
        plt.title('Random Forest')

        plt.subplot(1, 3, 2)
        sn.heatmap(cmGBMT,annot=True, cmap ='Spectral') # GB / Mobility
        plt.title('Gradient Boosting')

        plt.subplot(1, 3, 3)
        sn.heatmap(cmKNNMT,annot=True, cmap ='Spectral') # kNN / Mobility
        plt.title('K-Nearest Neighbors')
        plt.suptitle('Mobility feature of Target Stimuli')
        
        
        ##Visualization of confusion matrixes for Complexity parameters
        plt.figure(figsize=(13,4),  tight_layout=True)
        plt.subplot(1, 3, 1)
        sn.heatmap(cmRFCT,annot=True, cmap ='Spectral')# Random Forest / Complexity
        plt.title('Random Forest')

        plt.subplot(1, 3, 2)
        sn.heatmap(cmGBCT,annot=True, cmap ='Spectral') # GB / Complexity
        plt.title('Gradient Boosting')

        plt.subplot(1, 3, 3)
        sn.heatmap(cmKNNCT,annot=True, cmap ='Spectral') # kNN / Complexity
        plt.title('K-Nearest Neighbors')
        plt.suptitle('Complexity feature of Target Stimuli')
        plt.show()
        
    def plot_CM_distractor(self):
        
        """
        This function is used to plot the confusion matrices for the features of distractor stimuli. 
        It begins by averaging the list of confusion matrices obtained for each fold. By averaging
        these matrices, we obtain a consolidated representation of the overall performance of the
        classifiers on the standard stimuli features. The resulting averaged confusion matrix is 
        then visualized using the seaborn's heatmap. 
        """
        
        # Confuison matrix representing Random Forest results  
        cmRFAD = pd.concat(self.Results.mlAD.rf.get('cm')).groupby(level=0).mean().round(0)   # Activity
        cmRFMD = pd.concat(self.Results.mlMD.rf.get('cm')).groupby(level=0).mean().round(0) # Mobility
        cmRFCD = pd.concat(self.Results.mlCD.rf.get('cm')).groupby(level=0).mean().round(0) # Complexity
        
        # Confuison matrix representing Gradient Boosting results
        cmGBAD = pd.concat(self.Results.mlAD.gb.get('cm')).groupby(level=0).mean().round(0) # Activity
        cmGBMD = pd.concat(self.Results.mlMD.gb.get('cm')).groupby(level=0).mean().round(0) # Mobility
        cmGBCD = pd.concat(self.Results.mlCD.gb.get('cm')).groupby(level=0).mean().round(0) # Complexity
        
        # Confuison matrix representing Random kNN results
        cmKNNAD = pd.concat(self.Results.mlAD.knn.get('cm')).groupby(level=0).mean().round(0) # Activity
        cmKNNMD = pd.concat(self.Results.mlMD.knn.get('cm')).groupby(level=0).mean().round(0) # Mobility
        cmKNNCD = pd.concat(self.Results.mlCD.knn.get('cm')).groupby(level=0).mean().round(0) # Complexity

        
        # Visualization of confusion matrixes for activity parameters
        plt.figure(figsize=(13,4),  tight_layout=True)
        plt.subplot(1, 3, 1)
        sn.heatmap(cmRFAD,annot=True, cmap ='Spectral') # Random Forest / Activity
        plt.title('Random Forest')

        plt.subplot(1, 3, 2)
        sn.heatmap(cmGBAD,annot=True, cmap ='Spectral') # Gradient Boosting / Activity
        plt.title('Gradient Boosting')

        plt.subplot(1, 3, 3)
        sn.heatmap(cmKNNAD,annot=True, cmap ='Spectral')# kNN / Activity
        plt.title('K-Nearest Neighbors')
        plt.suptitle('Activity feature of Distractor Stimuli')

        #Visualization of confusion matrixes for Mobility parameters
        plt.figure(figsize=(13,4),  tight_layout=True)
        plt.subplot(1, 3, 1)
        sn.heatmap(cmRFMD,annot=True, cmap ='Spectral') # RF / Mobility
        plt.title('Random Forest')

        plt.subplot(1, 3, 2)
        sn.heatmap(cmGBMD,annot=True, cmap ='Spectral') # GB / Mobility
        plt.title('Gradient Boosting')

        plt.subplot(1, 3, 3)
        sn.heatmap(cmKNNMD,annot=True, cmap ='Spectral') # kNN / Mobility
        plt.title('K-Nearest Neighbors')
        plt.suptitle('Mobility feature of Distractor Stimuli')
        
        
        ##Visualization of confusion matrixes for Complexity parameters
        plt.figure(figsize=(13,4),  tight_layout=True)
        plt.subplot(1, 3, 1)
        sn.heatmap(cmRFCD,annot=True, cmap ='Spectral')# Random Forest / Complexity
        plt.title('Random Forest')

        plt.subplot(1, 3, 2)
        sn.heatmap(cmGBCD,annot=True, cmap ='Spectral') # GB / Complexity
        plt.title('Gradient Boosting')

        plt.subplot(1, 3, 3)
        sn.heatmap(cmKNNCD,annot=True, cmap ='Spectral') # kNN / Complexity
        plt.title('K-Nearest Neighbors')
        plt.suptitle('Complexity feature of Distractor Stimuli')
        plt.show()


