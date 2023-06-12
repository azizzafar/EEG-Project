# 1. Applying deep learning/ machine learning algorithms to classify patients using EEG Signals


# 2. Project Description
In this project, the task is to investigate and compare the performance of machine learning (ML), and deep learning (DL)
algorithms as classification techniques for event related potentials (ERPs) waveform. The focus is to analyze and
evaluate the effectiveness of ML, and DL algorithms in accurately classifying subjects
based on their ERPs patterns. By Exploring and comparing the performance of these
classification techniques, this thesis aims to contribute to the understanding and advancing
ERPs based classification methods in the field of neuroscience and cognitive research

# 3. How to Install and Run the Project
The code and dataset used for this thesis have been included as part of the "attachment.zip" file, which is provided alongside this thesis. The "attachment.zip" file contains all the necessary files and resources to reproduce and verify the results presented in this thesis. 

## 3.1 Install the necessary libraries
- pip install glob 
- pip install mne
- pip install pandas
- pip install numpy
- pip isntall seaborn
- pip install scikit-learn
- pip install h5py
- pip install keras
- pip install tensorflow
- pip install os-sys
- pip install PyWavelets
- pip install scikit-image


## 3.2 Raw
This file contains the raw EEG signal data for 91 patients, with 84 patients included in the study. Seven patients were excluded for reasons such as missing target stimuli annotation or subject group labels.  

## 3.3 preprocessed_data
This file contains the clean and sorted epochs of EEG signals ready for feature extraction and classification.

## 3.4 pasient_data
This Excel file contains information about the patients, and it's used to find the characteristics of the subject groups.

## 3.5 preprocessing.py
The "preprocessing.py" file includes a Python script that consists of two classes. The first class is responsible for reading, cleaning, and epoching the raw EEG data for each subject group individually. It contains methods or functions to perform data cleaning techniques, such as noise removal or artifact rejection, and epoching the data into segments of interest.

The second class utilizes the first class to read the data for each subject group, apply the necessary preprocessing steps using the methods from the first class, and save the preprocessed data into the "preprocessed_data" file as NumPy's ndarrays.![Alt text](preprocessing_steps-1.jpg)

This screenshot illustrates the process of reading the path storing EEG signal data and subsequently splitting the data based on subject groups. Each subject group's data is then assigned to a separate list for easy organization and analysis.

The lists are then fed into the second class in "preprocessing.py" for being preprocessed and saved into "preprocessed_data" file. 


## 3.6 Hjorths.py
The "Hjorths.py" file contains a Python script with two classes.
The first class takes the preprocessed data as input and performs computations to derive Hjorth's parameters, which are descriptors of the EEG signal's mobility, complexity, and activity. The class then returns a dataframe containing the calculated parameters for the respective subject group.

The second class utilizes the results obtained from the first class. It assigns labels to the subject groups and their orresponding classes. 
![Alt text](Load_compute_HP.JPG)

The screenshot demonstrates the process of loading preprocessed data and feeding them into the "hjorths" class for feature extraction. 



## 3.7 ML.py
The "ML.py" file consists of five classes that handle various classification and result analysis tasks.

The first class is responsible for classifying the subjects into groups based on a given dataset. It takes the dataset as input and applies a classification algorithm to assign subjects to specific groups. The class then returns the performance results, such as accuracy, f1-score, precision, recall, and confusion matrix.

The second class iterates through the datasets, applies the classification algorithm using the first class, and collects the results for easy access.

The third class is responsible for identifying classifiers with the highest performance based on metrics like f1-score, precision, and recall.

The fourth class highlights the best performance by accuracy using the accuracies obtained from the second class.

The fifth class is used for plotting confusion matrices.


![Alt text](classifications_results.JPG)


The screenshot illustrates the execution of classification and evaluation algorithms to obtain results. The second class in "ML.py" is employed to classify all datasets, enabling easy access and comparison of results. Subsequently, the third, fourth, and fifth classes are utilized to visualize the best performance and confusion matrices.




## 3.8 DL.py
The "DL.py" file is a class-based Python script that utilizes EEGNet for subject classification based on ERPs patterns. This class takes the epochs of raw EEG data as input. It applies the EEGNet model to classify the subjects and returns various performance metrics, including accuracy, f1-score, precision, recall, and a confusion matrix.

In addition to performance metrics, the class also provides visualization capabilities. It can also generate plots for accuracy and validation accuracy, along with loss and validation loss. 

![Alt text](DL_results.JPG)


# 3.9 Results.ipynb
By running the "Results.ipynb" Jupyter Notebook file, you can generate all the results presented in this thesis. The notebook contains the necessary code and instructions to load the data, apply preprocessing techniques, extract features, and perform classification using ML and DL algorithms.
