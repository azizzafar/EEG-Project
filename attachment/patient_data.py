#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd


# In[1]:


class patient_data:
    def __init__(self):
        
        '''
        This class generates the table containing the characteristics of the five
        subject groups using the provided Excel file "patient_data". The Excel file
        serves as the input data source for the relevant characteristics of each 
        subject group.  
        '''
        df = pd.read_excel(r'pasient_data.xls')
        df = df[['ID','Diagnosis', 'Sex', 'Age']]
        
        # Dropping the patients with lacking appropriate data
        df.drop(df.index[df['ID'] == 2], inplace = True) 
        df.drop(df.index[df['ID'] == 9], inplace = True)
        df.drop(df.index[df['ID'] == 81], inplace = True)
        df.drop(df.index[df['ID'] == 86], inplace = True)
        df.drop(df.index[df['ID'] == 91], inplace = True)
        df.drop(df.index[df['ID'] == 93], inplace = True)
        df.drop(df.index[df['ID'] == 6], inplace = True)
        df.drop(df.index[df['ID'] == 85], inplace = True)
        
        # Assigning the binary number to male/female class
        df.Sex[df.Sex == 1] = 'Male'
        df.Sex[df.Sex == 2] = 'Female'
        
        # Assigning the disease codes to the actual disease 
        df.Diagnosis[df.Diagnosis == 'F001'] = 'AD'
        df.Diagnosis[df.Diagnosis == 'F023'] = 'PDD'
        df.Diagnosis[df.Diagnosis == 'G20'] = 'PD'
        df.Diagnosis[df.Diagnosis == 'Kontroll'] = 'HC'
     

        # Filtering subject groups based on their disease
        self.df_DLB = df.query('Diagnosis== "DLB"')
        self.df_AD = df.query('Diagnosis== "AD"')
        self.df_PDD = df.query('Diagnosis== "PDD"')
        self.df_PD = df.query('Diagnosis== "PD"')
        self.df_HC = df.query('Diagnosis== "HC"')
        
        
    def group_data(self, df):
        '''
        This function gets the dataframe of a subject group, calculates mean 
        and standard deviation of their ages, and finds the number of male and
        females in the subject group. 
        
        param:   df: dataframe
        return:  df_patients: dataframe containing the the characteristic of
                              a subject group
        '''
        
        diagnosis = list(df.Diagnosis)  
        mean = int(round(df.Age.mean(), 0))
        std = round(df.Age.std(), 2)
        male = len(df.index[df['Sex'] =='Male'])
        female = len(df.index[df['Sex'] =='Female'])
        df_pasients = pd.DataFrame()
        df_pasients['Diagnosis'] = [diagnosis[0]]
        df_pasients['Sex(male/female)'] = str(male) +  f'/{female} '
        df_pasients['Age ± std'] = str(mean) + f' ± {std} '
        return df_pasients
    
    def patient_data(self):
        '''
        This all the five dataframe into one dataframe
        
        return: dataframe
        '''
        df_DLB = self.group_data(self.df_DLB)
        df_AD = self.group_data(self.df_AD)
        df_PDD = self.group_data(self.df_PDD)
        df_PD = self.group_data(self.df_PD)
        df_HC = self.group_data(self.df_HC)
        
        df = pd.concat([df_DLB, df_AD, df_PDD, df_PD, df_HC])
        df = df.reset_index(drop = True)
        return df


# In[ ]:




