import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import keras 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from keras.callbacks import Callback, EarlyStopping,ModelCheckpoint
from keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from keras import Input,Model
from keras.layers import Dense, Dropout, Activation ,Reshape, SpatialDropout2D, SeparableConv2D
from keras.layers import Conv2D, BatchNormalization,Flatten,MaxPooling1D, Lambda, DepthwiseConv2D,AveragePooling2D
from keras.constraints import max_norm
from keras import backend as K 

class EEGNet:
    def __init__(self, DLB, AD, PDD, PD, HC):
        
        """
        This class holds different functions, such as ensemble_data, EEGnet_model,
        getClassification, plots, clr_mean, clr_std, and mean_std. 
        The purpose of these functions are to produce the results by automat,
        without any need for change in parameters when we add more subject class,
        or remove a subject class. 
   
        params:   DLB: list of ndarray, or None
                  AD:  list of ndarray, or None
                  PDD: list of ndarray or None
                  PD:  list of ndarray or None
                  HC:  list of ndarray or None
        
        """
        
        self.DLB = DLB
        self.AD = AD
        self.PDD = PDD
        self.PD = PD
        self.HC = HC
        
        
        # list the input datas
        group_list = [self.DLB, self.AD, self.PDD, self.PD, self.HC]

        # remove the empty lists
        self.class_list = [] 
        for i in range(0, len(group_list)):
            if group_list[i] is not None:
                self.class_list.append(group_list[i])
            else:
                pass
            
        # features, labels and group-labels   
        self.X, self.y, self.groups = self.ensemble_data(self.class_list)
        
        # classification results produced by getPrediction function
        self.results = self.getClassification()
        
        
    def ensemble_data(self, subject_list):
        new_list = [] 
        for i in range(0, len(subject_list)):
            if subject_list[i] is not None:
                new_list.append(subject_list[i])
            else:
                pass

        label_list = []
        for i in range(0, len(new_list)):
            label_list.append([len(j)*[i] for j in new_list[i]])

        class_label = []
        for patient in label_list:
            for i in patient:
                class_label.append(i)


        # features
         # features
        features = []
        for subjects in new_list:
            if subjects is not None:
                subjects = list(subjects)
                features = features + subjects
        
        groups = [[i]*len(j) for i, j in enumerate(features)]
                
        return features, class_label, groups
    
    
    def EEGNet_model(self, Chans = 64, Samples = 128, 
             dropoutRate = 0.5, kernLength = 64, F1 = 8, 
             D = 2, F2 = 16, norm_rate = 0.25, dropoutType = 'Dropout'):
        
        """ 
        All the codes in this function are taken from the GITHUB of:
        ("Vernon J. Lawhern, Amelia J. Solon, Nicholas R. Waytowich, Stephen M. Gordon,
        Chou P. Hung, and Brent J. Lance. EEGNet: A compact convolutional neural
        network for EEG-based brain-computer interfaces. Journal of Neural Engineering,
        15(5), 7 2018. ISSN 17412552. doi: 10.1088/1741-2552/aace8c")
        
        link to the github: https://github.com/vlawhern/arl-eegmodels/blob/master/EEGModels.py
        
        USED UNDER THE LICENCE OF : Creative Commons Zero (CC0) License
        
        
        Keras Implementation of EEGNet
        http://iopscience.iop.org/article/10.1088/1741-2552/aace8c/meta

        Inputs:

          nb_classes      : int, number of classes to classify
          Chans, Samples  : number of channels and time points in the EEG data
          dropoutRate     : dropout fraction
          kernLength      : length of temporal convolution in first layer. We found
                            that setting this to be half the sampling rate worked
                            well in practice.     
          F1, F2          : number of temporal filters (F1) and number of pointwise
                            filters (F2) to learn. Default: F1 = 8, F2 = F1 * D. 
          D               : number of spatial filters to learn within each temporal
                            convolution. Default: D = 2
          dropoutType     : Either SpatialDropout2D or Dropout, passed as a string.

        """

        if dropoutType == 'SpatialDropout2D':
            dropoutType = SpatialDropout2D
        elif dropoutType == 'Dropout':
            dropoutType = Dropout
        else:
            raise ValueError('dropoutType must be one of SpatialDropout2D '
                             'or Dropout, passed as a string.')
            
        input1   = Input(shape = (Chans, Samples, 1))

        ##################################################################
        block1       = Conv2D(F1, (1, kernLength), padding = 'same',
                                       input_shape = (Chans, Samples, 1),
                                       use_bias = False)(input1)
        block1       = BatchNormalization()(block1)
        block1       = DepthwiseConv2D((Chans, 1), use_bias = False, 
                                       depth_multiplier = D,
                                       depthwise_constraint = max_norm(1.))(block1)
        block1       = BatchNormalization()(block1)
        block1       = Activation('elu')(block1)
        block1       = AveragePooling2D((1, 4))(block1)
        block1       = dropoutType(dropoutRate)(block1)

        block2       = SeparableConv2D(F2, (1, 16),
                                       use_bias = False, padding = 'same')(block1)
        block2       = BatchNormalization()(block2)
        block2       = Activation('elu')(block2)
        block2       = AveragePooling2D((1, 8))(block2)
        block2       = dropoutType(dropoutRate)(block2)

        flatten      = Flatten(name = 'flatten')(block2)

        dense        = Dense(len(self.class_list), name = 'dense', 
                             kernel_constraint = max_norm(norm_rate))(flatten)
        softmax      = Activation('softmax', name = 'softmax')(dense)
        model = Model(inputs=input1, outputs=softmax)
        model.compile(loss = "categorical_crossentropy", optimizer = Adam(lr = 0.0005), metrics = ["accuracy"])

        return model
    def getClassification(self):
        
        """
        This function utilizes input datasets and the EEGNet model
        function to perform subject classification. The process involves the 
        following steps:
        *   Dataset Splitting: The datasets are divided into 5 folds using the
            groupKfold function from scikit-learn. This ensures that data from a
            particular subject is not present in both the training and validation 
            sets. This is important to avoid potential bias and to evaluate the 
            model's generalization ability.
            
        *   Training and Testing: The function trains the model on four of the folds
            and then tests it on the remaining fold. This process is repeated for 
            each fold, resulting in a comprehensive evaluation across all folds.
            
        *   Performance Metrics: During testing, the function calculates various 
            performance metrics such as accuracy, f1-score, precision, and recall.
            These metrics provide an assessment of the model's classification 
            performance, allowing for a quantitative evaluation of its effectiveness.
        """
        # samples and channels shapes, used for input_shape for the model
        samples, channels  = self.X[0].shape[2], self.X[0].shape[1]
   
        X  = np.vstack(self.X)
        #X = np.moveaxis(X,1,2)

        y = np.hstack(self.y)
        groups = np.hstack(self.groups)
  
        gkf = GroupKFold(n_splits = 5)
    
        history_list = []
        accuracy_list = []
        cm_list = []
        clr_list = []
        ## splitting data using groupKFold/GroupShuffleSplit
        for train_index, test_index in gkf.split(X, y, groups = groups):
            X_train, y_train, g_train = X[train_index], y[train_index], groups[train_index]
            X_test,  y_test, g_test  = X[test_index],  y[test_index], groups[test_index]
            y_true = y_test

            # scaling training and sience to get the zero mean
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train.reshape(-1, 
                                            X_train.shape[-1])).reshape(X_train.shape)
            X_test = scaler.transform(X_test.reshape(-1, 
                                            X_test.shape[-1])).reshape(X_test.shape)
            X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2], 1))
            
             ## convert the target variable to one-hot encoded categorical data.
            y_train = keras.utils.to_categorical(y_train , num_classes= None)
            y_test = keras.utils.to_categorical( y_test, num_classes=None)
            
            
            # Create checkpoint callback
            mc = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
            
            # Setup EarlyStopping callback to stop training if model's loss doesn't decreases for 5 epochs 
            es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience= 5)

            # initiate the model
            model = self.EEGNet_model(Chans = channels, Samples = samples, 
                                     dropoutRate = 0.5, kernLength = 64, F1 = 4, 
                                     D = 2, F2 = 8, norm_rate = 0.25, dropoutType = 'Dropout')
            # fit the model with data
            history = model.fit( X_train, y_train, epochs = 100, callbacks = [es, mc],
                                batch_size = 128, validation_data=(X_test, y_test))
            
            history_list.append(history)

            ## save the best model
            #best_model = tf.keras.models.load_model('best_model.h5')
            y_pred     = model.predict(X_test)
            y_pred     = np.argmax(y_pred, axis=1)
            #accuracy   = best_model.evaluate(X_val, y_val)
            accuracy = model.evaluate(X_test, y_test)[1]
            accuracy_list.append(accuracy)
            
            clr_list.append(pd.DataFrame(classification_report(y_true, y_pred, 
                                                          output_dict = True, zero_division = 0)))

            df = pd.DataFrame({'y_true': y_true, 
                               'y_pred': y_pred})
            cm_list.append(pd.crosstab(df.y_true, df.y_pred))
        
        return {'history':history_list,'clr': clr_list, 'cm': cm_list, 'accuracy':accuracy_list}
    
    
    def plots(self):
        
        """
        This function plots the accuracy, validation accuracy, loss and validation loss,
        produced by the "getClassification" function.
        """
        
        # load the history 
        history = self.results.get('history')
        
        # plot the results for the first fold
        epochs1 = range(len(history[0].history['accuracy']))  
        fig, axs = plt.subplots(2, 2, figsize=(15, 15))
        axs[0, 0].plot(epochs1, history[0].history['accuracy'], label='Training accuracy')
        axs[0, 0].plot(epochs1, history[0].history['val_accuracy'], label='validation accuracy')
        axs[0, 0].plot(epochs1, history[0].history['loss'], label = 'Training loss')
        axs[0, 0].plot(epochs1, history[0].history['val_loss'], label = 'Validation loss')
        axs[0, 0].set_title('Fold-0')
        axs[0, 0].legend(loc=7)

        # plot the results for the second fold
        epochs2 = range(len(history[1].history['accuracy'])) 
        axs[0, 1].plot(epochs2, history[1].history['accuracy'], label ='Training accuracy' )
        axs[0, 1].plot(epochs2, history[1].history['val_accuracy'], label='validation accuracy')
        axs[0, 1].plot(epochs2, history[1].history['loss'],label = 'Training loss')
        axs[0, 1].plot(epochs2, history[1].history['val_loss'], label = 'Validation loss')
        axs[0, 1].set_title('Fold-1')
        axs[0, 1].legend(loc=7)

        # plot the results for the third fold
        epochs3 = range(len(history[2].history['accuracy'])) 
        axs[1, 0].plot(epochs3, history[2].history['accuracy'], label ='Training accuracy')
        axs[1, 0].plot(epochs3, history[2].history['val_accuracy'], label='validation accuracy')
        axs[1, 0].plot(epochs3, history[2].history['loss'], label = 'Training loss')
        axs[1, 0].plot(epochs3, history[2].history['val_loss'], label = 'Validation loss')
        axs[1, 0].set_title('Fold-2')
        axs[1, 0].legend(loc=7)

        # plot the results for the fourth fold
        epochs3 = range(len(history[2].history['accuracy'])) 
        epochs4 = range(len(history[3].history['accuracy'])) 
        axs[1, 1].plot(epochs4, history[3].history['accuracy'], label ='Training accuracy')
        axs[1, 1].plot(epochs4, history[3].history['val_accuracy'], label='validation accuracy')
        axs[1, 1].plot(epochs4, history[3].history['loss'], label = 'Training loss')
        axs[1, 1].plot(epochs4, history[3].history['val_loss'], label = 'Validation loss')
        axs[1, 1].set_title('Fold-3')
        axs[1, 1].legend(loc=7)
        plt.show()
        
    
    def clr_mean(self):
        
        """
        This function use the results of classification report produced by
        "getClassification" and find the mean for f1-scores, precisions, and recalls
        of all the folds. 
        
        return: dataframe, containing f1-score, precision, and recall for the subject groups.
        """
        df = pd.concat(self.results.get('clr')).groupby(level=0).mean()[:3]
        approachs = ['EEGNet', 'EEGNet', 'EEGNet']
        dataset = ['Standard','Standard','Standard']

        df = df.T[:5]
        df = df.T
        col = ['DLB', 'AD', 'PDD', 'PD', 'HC']
        df.columns = col        
        df['Approach'] = approachs
        df['Dataset'] = dataset
        return df
    def clr_std(self):
        """
        This function use the results of classification report produced by
        "getClassification" and find the standard devidation for f1-scores, precisions, 
        and recalls of all the folds. 
        
        return: dataframe, containing f1-score, precision, and recall for the subject groups.
        """
        df = pd.concat(self.results.get('clr')).groupby(level=0).std()[:3]
        approachs = ['EEGNet', 'EEGNet', 'EEGNet']
        dataset = ['Standard','Standard','Standard']

        df = df.T[:5]
        df = df.T
        col = ['DLB', 'AD', 'PDD', 'PD', 'HC']
        df.columns = col        
        df['Approach'] = approachs
        df['Dataset'] = dataset
        return df
    def mean_std(self):
        dfmean = self.clr_mean()
        dfstd  = self.clr_std()
        df_mean_std = pd.DataFrame()
        df_mean_std['AD'] = (dfmean['AD'].round(decimals = 3).astype('str')
                             + ' ( ' + dfstd['AD'].round(decimals = 2).astype('str') + ')')
        df_mean_std['DLB'] = (dfmean['DLB'].round(decimals = 3).astype('str')
                              + ' ( ' + dfstd['DLB'].round(decimals = 2).astype('str') + ')')
        df_mean_std['HC'] = (dfmean['HC'].round(decimals = 3).astype('str')
                             + ' ( ' + dfstd['HC'].round(decimals = 2).astype('str') + ')')
        df_mean_std['PD'] = (dfmean['PD'].round(decimals = 3).astype('str')
                             + ' ( ' + dfstd['PD'].round(decimals = 2).astype('str') + ')')
        df_mean_std['PDD'] = (dfmean['PDD'].round(decimals = 3).astype('str')
                              + ' ( ' + dfstd['PDD'].round(decimals = 2).astype('str') + ')')
        
        df_mean_std['Dataset'] = (dfmean['Dataset'])
        return df_mean_std
    
    def clr_mean_2class(self):

        """
        This function use the results of classification report produced by
        "getClassification" and find the mean for f1-scores, precisions, and recalls
        of all the folds for 2-class classification problem. 

        return: dataframe, containing f1-score, precision, and recall for the subject groups.
        """
        df = pd.concat(self.results.get('clr')).groupby(level=0).mean()[:3]
        df = df.T[:5]
        df = df.T
        col = ['class1', 'HC', 'Accuracy', 'Macro avg', 'Weiwghted avg']
        df.columns = col        
        return df

    def clr_std_2class(self):
        """
        This function use the results of classification report produced by
        "getClassification" and find the standard devidation for f1-scores, precisions, 
        and recalls of all the folds for 2-class classification problem. 

        return: dataframe, containing f1-score, precision, and recall for the subject groups.
        """
        df = pd.concat(self.results.get('clr')).groupby(level=0).std()[:3]

        df = df.T[:5]
        df = df.T
        col = ['class1', 'HC', 'Accuracy', 'Macro avg', 'Weiwghted avg']
        df.columns = col        
        return df

    def mean_std_2class(self):
        dfmean = self.clr_mean_2class()
        dfstd  = self.clr_std_2class()
        df_mean_std = pd.DataFrame()
        df_mean_std['class1'] = (dfmean['class1'].round(decimals = 3).astype('str')
                                 + ' ( ' + dfstd['class1'].round(decimals = 2).astype('str') + ')')
        df_mean_std['HC'] = (dfmean['HC'].round(decimals = 3).astype('str') 
                             + ' ( ' + dfstd['HC'].round(decimals = 2).astype('str') + ')')
        df_mean_std['Accuracy'] = (dfmean['Accuracy'].round(decimals = 3).astype('str') 
                                   + ' ( ' + dfstd['Accuracy'].round(decimals = 2).astype('str') + ')')
        df_mean_std['Macro avg'] = (dfmean['Macro avg'].round(decimals = 3).astype('str')
                                    + ' ( ' + dfstd['Macro avg'].round(decimals = 2).astype('str') + ')')
        df_mean_std['Weiwghted avg'] = (dfmean['Weiwghted avg'].round(decimals = 3).astype('str')
                                        + ' ( ' + dfstd['Weiwghted avg'].round(decimals = 2).astype('str') + ')')
                                  
        return df_mean_std

    def mean_std_results(self):
        if len(self.class_list) == 2:
            return self.mean_std_2class()
        elif len(self.class_list) == 5:
            return self.mean_std()

