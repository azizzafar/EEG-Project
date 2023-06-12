#!/usr/bin/env python
# coding: utf-8

# In[15]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import matplotlib.patches as mpatches
import seaborn as sn
import matplotlib.pyplot as plt


# In[16]:


class hjorths:
    def __init__(self, data, average = False):
        """
        This class takes the data for one of the subject groups as input
        and computes Hjorth's parameters for the provided data. Once computed,
        these parameters are organized and stored in dataframes for further analysis. 
        
        parameter:
                data: list of ndarray 
                average: False|True
                    If True: all the channels will be averaged into one channel
                    IF False: ndarray
                              
        
        """
        self.data = data
        self.average = average
        
        
        # Load the results of "getHjorthsParameters" function for easy access.
        self.hjorth_parameters = self.getHjorthsParameters(self.data, average = self.average)

        # Load the resutls of "extract_parameters" for easy access.
        self.a, self.m, self.c = self.extract_parameters(self.hjorth_parameters)
        
        # Channel names for the two scenarios initially used in this project.
        # 1. None of the channels dropped, in which we can use the first list of 
        # channel names as index for the dataframe of Hjorth's parameters.
        
        # 2. Dropping noisy channels and working with the remaining 9-channels. 
        # In this case the the shorter list of the channel names are used as 
        # index for the Hjorth's parameters. 
        if self.data[0].shape[1] == 16:
            self.ch_names = ['V-EOG',
                                 'H-EOG',
                                 'P3',
                                 'Pz',
                                 'P4',
                                 'T7',
                                 'T8',
                                 'O1',
                                 'Fp1',
                                 'Fp2',
                                 'F3',
                                 'Fz',
                                 'F4',
                                 'C3',
                                 'Cz',
                                 'C4']
        else:
            self.ch_names = ['P3',
                             'Pz',
                             'P4',
                             'F3',
                             'Fz',
                             'F4',
                             'C3',
                             'Cz',
                             'C4']
        
        ###################################################################################################
    def getHjorthsParameters(self, dataset, average = False):
        
        """
        This function receives the dataset, which is a list of ndarrays, for a subject
        group as input. It then proceeds to calculate Hjorth's parameters for the given dataset. 

        
        parameters:
                1. dataset: the dataset for one of the subject groups in the shape of
                   (patient, epochs, channel, sample).
                2. average: averages the dataset across channels if asked for.
                
        return: 
                parameters: Hjorts parameters as a list of ndarray.
        
        """
        
        parameters = []
        for data in dataset:
            
            activity   = []
            mobility   = []
            complexity = []
            if type(data) == np.ndarray:
                if average == True:
                    data = np.mean(data, axis =1)
                    for signal in data:
                        # compute first derivative of signal
                        diff1 = np.diff(signal)
                        # compute second derivative of signal 
                        diff2 = np.diff(signal, 2)
                        ## compute variance of signal
                        var_signal = np.var(signal) 
                        # compute standard deviation of the signal and derivatives
                        std_signal = np.std(signal)
                        std_diff1   = np.std(diff1)
                        std_diff2  = np.std(diff2)

                        ## compute mobility 
                        mob = np.sqrt(std_diff1/std_signal)

                        ## compute complexity 
                        comp= np.sqrt(std_diff2 / std_diff1) / mob


                        activity.append(var_signal)
                        mobility.append(mob)
                        complexity.append(comp)
                else:
                    for signal in data:
                        for epochs in signal:
                            # compute first derivative of signal
                            diff1 = np.diff(epochs)
                            # compute second derivative of signal 
                            diff2 = np.diff(epochs, 2)
                            
                            ## compute variance of signal
                            var_epochs = np.var(epochs) 
                            
                            # compute standard deviation of the signal and derivatives
                            std_epochs = np.std(epochs)
                            std_diff1   = np.std(diff1)
                            std_diff2  = np.std(diff2)

                            ## compute mobility 
                            mob = np.sqrt(std_diff1/std_epochs)

                            ## compute complexity 
                            comp= np.sqrt(std_diff2 / std_diff1) / mob

                            activity.append(var_epochs)
                            mobility.append(mob)
                            complexity.append(comp)


            else:
                pass
            
            if average == False:
                reshaped_parameters = (np.array(activity).reshape(np.array(data).shape[0], np.array(data).shape[1] ),
                        np.array(mobility).reshape(np.array(data).shape[0], np.array(data).shape[1] ),
                        np.array(complexity).reshape(np.array(data).shape[0], np.array(data).shape[1]))
                parameters.append(reshaped_parameters)
            else:
                
                param = (activity, mobility, complexity)
                parameters.append(param)
                
        return parameters
    
    
    #######################################################################################################        
    #return activity, mobility, complexity 
    # extract parameters from eachother. 
    def extract_parameters(self, hjorth_parameters):
        """
        parameter: 
                hjorth_parameters:   list of arrays (results from the "getHjorthsParameters")
        
        
        return: 
                activity:   list of array (activity parameter of Hjorth's parameters).
                            Each array in the list contain the activity parameters for one subject. 
                mobility:   list of array (mobility parameter of Hjorth's parameters).
                            Each array in the list contain the mobility parameters for one subject. 
                complexity: list of array (complexity parameter of Hjorth's parameters).
                            Each array in the list contain the complexity parameters for one subject. 
        """
        activity  = []
        mobility   = []
        complexity = []
        for features in hjorth_parameters:
            a, b, c = features
            activity.append(a)
            mobility.append(b)
            complexity.append(c)
        return activity, mobility, complexity
    
    ########################################################################
    def minMax_Scaler(self, df):
        scaler = MinMaxScaler(feature_range = (0, 1))
        scaler.fit(df)
        scaled_df = scaler.transform(df)
        return scaled_df
    
    ################################################################################################
    def getDataFrameActivity(self):
        
        """
        This function converts the activity parameter for a group
        into a dataframe and scaling it between 0 and 1.
        
        return: dataframe, containing activity parameters for a group. 
        """
        
        if self.average == True:
            df = pd.DataFrame(self.a)
        else:
            df = [pd.DataFrame(i.T, index = self.ch_names) for i in self.a]
            df = pd.concat(df)
        index = df.index 
        #df = self.standard_Scaler(df)
        df = self.minMax_Scaler(df)
        df = pd.DataFrame(df, index = index)
        return df
    
    ###############################################################################################
    def getDataFrameMobility(self):
        
        """
        This function converts the mobility parameter for a group
        into a dataframe and scaling it between 0 and 1.
        
        return: dataframe, containing mobility parameters for a group. 
        """
        if self.average == True:
            df = pd.DataFrame(self.m)
        else:
            df = [pd.DataFrame(i.T, index = self.ch_names) for i in self.m]
            df = pd.concat(df)
        index = df.index
        #df = self.standard_Scaler(df)
        df = self.minMax_Scaler(df)
        df = pd.DataFrame(df, index = index)
        return df
    
    ################################################################################################
    def getDataFrameComplexity(self):
        
        """
        This function converts the complexity parameter for a group
        into a dataframe and scaling it between 0 and 1.
        
        return: dataframe, containing complexity parameters for a group. 
        """
        if self.average == True:
            df = pd.DataFrame(self.a)
        else:
            df = [pd.DataFrame(i.T, index = self.ch_names) for i in self.c]
            df = pd.concat(df)
        index = df.index
        #df = self.standard_Scaler(df)
        df = self.minMax_Scaler(df)
        df = pd.DataFrame(df, index = index)
        return df 


# In[17]:


class getFeaturesIntoDataframe:
    def __init__(self, DLB = None, AD = None, PDD = None, PD = None, HC = None,num_channels = None, 
                 Activity = False, Mobility = False, Complexity = False, bundle = False):
        
        
        """
        TThis class uses the "hjorths" class to compute Hjorth's parameters for all subject groups.
        After calculating the parameters, it assigns labels to the subject groups and their corresponding
        classes. The class then returns the complete dataset through the "getDataset" function, which 
        combines the "getGroupFeaturesIntoDataframe", "getLabel", and "groupLabel" functions.
        
        Initially, this class was designed to handle two or more subject-class pairs as input and
        assign labels to the subject groups and their corresponding classes. Therfore the default
        values for the subject groups were set to None.  The intention was to analyze and evaluate
        the classification results between two groups at a time. However, since the script is still
        capable of handling the final decision of multiclass classification, no modifications were
        made to restrict it to only multiclass classification tasks
        
        parameters: 
                DLB: class, (class "hjorths" run on the data for the DLB subject group to compute activity, 
                     mobility and complexity parameters).
                     
                AD:  class, (class "hjorths" run on the data for the AD subject group to compute activity, 
                     mobility and complexity parameters).
                     
                PDD: class, (class "hjorths" run on the data for the PDD subject group to compute activity, 
                     mobility and complexity parameters).
                     
                PD:  class, (class "hjorths" run on the data for the PD subject group to compute activity, 
                     mobility and complexity parameters).
                     
                HC:  class, (class "hjorths" run on the data for the HC subject group to compute activity, 
                     mobility and complexity parameters).
                     
                num_channel: number of channels per subject, so it can be assigned to the right group.
                
                Activity:   False|True
                        If True, the complete dataset for the activity parameter will be returned
                        when calling the "getDataset" function. 
                  
                Mobility:   False|True
                        If True, the complete dataset for the mobility parameter will be returned
                        when calling the "getDataset" function. 
                        
                Complexity: False|True
                        If True, the complete dataset for the complexity parameter will be returned
                        when calling the "getDataset" function. 
                        
                bundle:     False|True
                        If True, the complete datasets for Activity, Mobility and Complexity will be bundled
                        into one dataset when calling the "getDataset" function. 
                
        
        """
        
        self.DLB        = DLB
        self.AD         = AD
        self.PDD        = PDD
        self.PD         = PD
        self.HC         = HC
        self.num_channels = num_channels
        self.activity   = Activity
        self.mobility   = Mobility
        self.complexity = Complexity
        self.bundle = bundle 
        #self.dataset    = self.getDataset() 
    
    def getGroupFeaturesIntoDataframe(self):
        """
        This function use the activity, mobility and complexity parameters computed by the class "hjorths",
        and use the "getLabel", functions to assign class-label to them. 
        
        
        return: 
            df: dataframe
                If bundle == True: a dataframe of the complete dataset containing all the parameters
                will be returned when this function is called. 

                If bundle == False: a dataframe containing three dataframe will be returned. (dataframe for 
                activity, mobility and complexity parameters) 

        """
        df = []
        if self.DLB is not None:    
            DLB = self.DLB
            if self.bundle == True:
                Activity   = DLB.getDataFrameActivity().T
                Mobility   = DLB.getDataFrameMobility().T
                Complexity = DLB.getDataFrameComplexity().T
                
                # Bundle the hjort's parameters into one bundle 
                df_bundle = pd.concat([Activity, Mobility, Complexity]).T
                df_bundle = df_bundle.replace({np.NAN: 0})
                df_bundle = df_bundle.loc[:, df_bundle.isin(['NaN','NULL',0]).mean() < .80]
                df_bundle = df_bundle.T
                df_bundle = df_bundle.reset_index(drop=True).T
                #df_bundle = df_bundle.T
                label = self.getLabel(len(df_bundle),'DLB')
                df_bundle['label'] = label
                df.append(df_bundle)
            else:
                
                Activity   = DLB.getDataFrameActivity()
                label1 = self.getLabel(len(Activity),'DLB')
                Activity['label'] = label1

                Mobility   = DLB.getDataFrameMobility()
                label2 = self.getLabel(len(Mobility),'DLB')
                Mobility['label'] = label2

                Complexity = DLB.getDataFrameComplexity()
                label3 = self.getLabel(len(Complexity),'DLB')
                Complexity['label'] = label3

                list1 = {'Activity'   : Activity, 
                         'Mobility'   : Mobility,
                         'Complexity' : Complexity}
                dfDLB = pd.concat(list1).T
                df.append(dfDLB)
        else:
            pass
        
        if self.AD is not None:
            AD = self.AD
            if self.bundle == True:
                Activity   = AD.getDataFrameActivity().T
                Mobility   = AD.getDataFrameMobility().T
                Complexity = AD.getDataFrameComplexity().T
                
                # Bundle the hjort's parameters into one bundle 
                df_bundle = pd.concat([Activity, Mobility, Complexity]).T
                df_bundle = df_bundle.replace({np.NAN: 0})
                df_bundle = df_bundle.loc[:, df_bundle.isin(['NaN','NULL',0]).mean() < .80]
                df_bundle = df_bundle.T
                df_bundle = df_bundle.reset_index(drop=True).T
                #df_bundle = df_bundle.T
                label = self.getLabel(len(df_bundle),'AD')
                df_bundle['label'] = label
                df.append(df_bundle)
            else:
                Activity   = AD.getDataFrameActivity()
                label1 = self.getLabel(len(Activity),'AD')
                Activity['label'] = label1

                Mobility   = AD.getDataFrameMobility()
                label2 = self.getLabel(len(Mobility),'AD')
                Mobility['label'] = label2

                Complexity = AD.getDataFrameComplexity()
                label3 = self.getLabel(len(Complexity),'AD')
                Complexity['label'] = label3

                list2 = {'Activity'   : Activity, 
                         'Mobility'   : Mobility,
                         'Complexity' : Complexity}
                dfAD = pd.concat(list2).T
                df.append(dfAD)
        else:
            pass
        
        if self.PDD is not None:
            PDD  = self.PDD
            if self.bundle == True:
                Activity   = PDD.getDataFrameActivity().T
                Mobility   = PDD.getDataFrameMobility().T
                Complexity = PDD.getDataFrameComplexity().T
                
                # Bundle the hjort's parameters into one bundle 
                df_bundle = pd.concat([Activity, Mobility, Complexity]).T
                df_bundle = df_bundle.replace({np.NAN: 0})
                df_bundle = df_bundle.loc[:, df_bundle.isin(['NaN','NULL',0]).mean() < .80]
                df_bundle = df_bundle.T
                df_bundle = df_bundle.reset_index(drop=True).T
                #df_bundle = df_bundle.T
                label = self.getLabel(len(df_bundle),'PDD')
                df_bundle['label'] = label
                df.append(df_bundle)
            else:
                Activity   = PDD.getDataFrameActivity()
                label1 = self.getLabel(len(Activity),'PDD')
                Activity['label'] = label1

                Mobility   = PDD.getDataFrameMobility()
                label2 = self.getLabel(len(Mobility),'PDD')
                Mobility['label'] = label2

                Complexity = PDD.getDataFrameComplexity()
                label3 = self.getLabel(len(Complexity),'PDD')
                Complexity['label'] = label3

                list3      = {'Activity'   : Activity, 
                              'Mobility'   : Mobility,
                              'Complexity' : Complexity}
                dfPDD = pd.concat(list3).T
                df.append(dfPDD)
        
        if self.PD is not None:
            PD = self.PD
            
            if self.bundle == True:
                Activity   = PD.getDataFrameActivity().T
                Mobility   = PD.getDataFrameMobility().T
                Complexity = PD.getDataFrameComplexity().T
                
                # Bundle the hjort's parameters into one bundle 
                df_bundle = pd.concat([Activity, Mobility, Complexity]).T
                df_bundle = df_bundle.replace({np.NAN: 0})
                df_bundle = df_bundle.loc[:, df_bundle.isin(['NaN','NULL',0]).mean() < .80]
                df_bundle = df_bundle.T
                df_bundle = df_bundle.reset_index(drop=True).T
                #df_bundle = df_bundle.T
                label = self.getLabel(len(df_bundle),'PD')
                df_bundle['label'] = label
                df.append(df_bundle)
            else:        
                Activity   = PD.getDataFrameActivity()
                label1 = self.getLabel(len(Activity),'PD')
                Activity['label'] = label1

                Mobility   = PD.getDataFrameMobility()
                label2 = self.getLabel(len(Mobility),'PD')
                Mobility['label'] = label2

                Complexity = PD.getDataFrameComplexity()
                label3 = self.getLabel(len(Complexity),'PD')
                Complexity['label'] = label3

                list4      = {'Activity'   : Activity, 
                              'Mobility'   : Mobility,
                              'Complexity' : Complexity}
                dfPD = pd.concat(list4).T
                df.append(dfPD)

        else:
            pass
        
        if self.HC is not None:
            HC = self.HC
            if self.bundle == True:
                Activity   = HC.getDataFrameActivity().T
                Mobility   = HC.getDataFrameMobility().T
                Complexity = HC.getDataFrameComplexity().T
                
                # Bundle the hjort's parameters into one bundle 
                df_bundle = pd.concat([Activity, Mobility, Complexity]).T
                df_bundle = df_bundle.replace({np.NAN: 0})
                df_bundle = df_bundle.loc[:, df_bundle.isin(['NaN','NULL',0]).mean() < .80]
                df_bundle = df_bundle.T
                df_bundle = df_bundle.reset_index(drop=True)
                df_bundle = df_bundle.T
                label = self.getLabel(len(df_bundle),'HC')
                df_bundle['label'] = label
                df.append(df_bundle)
            else:
                Activity   = HC.getDataFrameActivity() # DataFrame 
                label1 = self.getLabel(len(Activity),'HC')
                Activity['label'] = label1

                Mobility   = HC.getDataFrameMobility()
                label2 = self.getLabel(len(Mobility),'HC')
                Mobility['label'] = label2

                Complexity = HC.getDataFrameComplexity()
                label3 = self.getLabel(len(Complexity),'HC')
                Complexity['label'] = label3

                list5 = {'Activity'   : Activity, 
                         'Mobility'   : Mobility,
                         'Complexity' : Complexity}
                dfHC = pd.concat(list5).T
                df.append(dfHC)
        else:
            pass
        
        return df
    ############################################################
    def getLabel(self, dataset, label):
        """
        This function is used to create class label for a subject group.
        
        parameters: 
                dataset: dataframe, dataframe for a subject group.
                label:   label, the name of the subject group or a number. 
            
        return: list, a list of the same lenght as the rows of the given 
                dataset with the same group name. 
        """
        labels = []
        for i in range (0, dataset):
            labels.append(label)
        return labels

        ########################################################
    def groupLabel(self, dataset):
        """
        This function is used to create group label for the complete dataset. 
        
        parameters:
                  dataset: dataframe, the complete dataset containing all the subject groups.
                  
        return: 
                  labels: group labels which is unique for each subject.            
        
        """
        n_channels = int(len(dataset)/self.num_channels)
        labels = []
        for i in range(0, n_channels):
            labels.append(self.getLabel(self.num_channels, i))
        return labels
        
    def getDataset(self):
        
        """
        This function use the "groupLabel" function to assign group-label to the dataframes produced by the 
        "getGroupFeaturesIntoDataframe". 
        
        
        return: 
                df: dataframe. The final dataframe that can be used for classification.
        
        """
        
        
        dataset = self.getGroupFeaturesIntoDataframe()
        df = []
        for data in dataset:
            if self.bundle == True:
                df.append(data)
            else:
                if self.activity == True:
                    activity = data.Activity
                    df.append(activity.T)
                    
                elif  self.mobility == True:
                    mobility = data.Mobility
                    df.append(mobility.T)
                    
                elif self.complexity == True:
                    complexity = data.Complexity
                    df.append(complexity.T)
                else:
                    pass 
                
        df = pd.concat(df)
        #df = df.dropna(axis =1)
        df = df.replace({np.NAN: 0})
        df = df.loc[:, df.isin(['NaN','NULL',0]).mean() < .80]
        
        group_label = self.groupLabel(df)
        group_label = [item for sublist in group_label for item in sublist]
        
        df['group_label'] = group_label
        return df 



