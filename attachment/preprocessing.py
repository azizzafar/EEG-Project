#!/usr/bin/env python
# coding: utf-8

# In[1]:


import glob
import os
import mne 
import numpy as np
import pandas as pd
from mne.preprocessing import ICA, EOGRegression
import pywt
from skimage.restoration import denoise_wavelet



class preprocessing:
    #################################################################################
    def __init__(self, path, apply_wavelet = False, apply_notch = False):
        
        """
        This class is responsible for reading, cleaning, and epoching the raw EEG data
        for each subject group individually.
        
        Most of the algorithms used in this class are taken from MNE, a Python package
        specifically designed for processing, analysis, and visualization of 
        neurophysiological data such as magnetoencephalography (MEG) and 
        electroencephalography (EEG) signals.
        URL: https://mne.tools/stable/index.html
    
        
        parameters:
            path:    list, list of path-like, ndarray, raw_eeglab
                
            apply_wavelet: False|True
                If True: The data will be denoised by wavelet denoising.
                If False: The data will not be denoised by wavlet.
                
            apply_notch: False|True
                If True: The data will be denoised by wavelet Nothch Filter.
                If False: The data will not be denoised by Nothch Filter.
        """
        
        self.path            = path      
        self.apply_wavelet   = apply_wavelet
        self.apply_notch     = apply_notch
        
        
        #  Load the results of "read_raw" fnction to make it accessible.
        self.dataset = self.read_raw(self.path)
        
        # Load the reuslts of "getStimuli" function to make it accessible. 
        self.stimuli =  self.getStimuli(dataset = self.dataset )

    #################################### Read, Clean and Epoch #############################################
    # read the path/dataset
    def read_raw(self, path):
        
        """
        
        This function can read different types of EEG data, including raw eeg_lab,
        epoched eeg_lab, and MNE files. If the data is of type raw_eeglab, the
        function will apply the following preprocessing steps:
       
            a. A bandpass filter will be used to remove frequencies outside 
             the EEG bands. 
            b. The average of all channels will be used as the reference. 
            c. V-EOG and H-EOG channels will be converted from EEG to EOG channels. 
            d. Regression techniques are applied to repair eye-movement artifacts. 
            e. ICA will be performed to remove noise and artifacts further.
      
            f. The data will then be epoched using annotations. 
            g. Baseline correction will be applied to the epochs. 
            h. Epochs violating the ±120 microvolt criterion will be rejected. 
            i. Channels considered too noisy are dropped.
            j. Lastly, the data will be resampled to 128 Hz. 
        These preprocessing steps ensure that the data is clean and ready for 
        further analysis.
        
        
        Parameters: 
                path:     path, list of ndarray, list of mne files.
        
        return: 
            list of ndarray, mne.Epochs.Epochs
        
        """
        dataset = [] 
        for data in path:
            if type(data) == np.ndarray:
                data = data
                dataset.append(data)

            elif type(data) == mne.evoked.EvokedArray:
                data = data.get_data()
                dataset.append(data)

            elif type(data) == mne.epochs.Epochs:
                epochs = data
                ica = ICA(n_components=None, method='picard',fit_params=dict(extended=True),
                              random_state=21, max_iter = 10000 )
                # Apply ICA 
                ica.fit(epochs)
                epochs = ica.apply(epochs)
                
                dataset.append(epochs)

            else:
            
                raw = mne.io.read_raw_eeglab(data)
                # filter out the frequencies outside the EEG bands.
                raw.filter(l_freq = 0.5, h_freq = 50 )
                # use the average of the channels as reference.
                raw.set_eeg_reference('average')
                # find the correct names of the channels
                new_ch_names = ['V-EOG',
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

                # convert the channels names from numbers to their real name by 10-20 system.
                old_ch_names = raw.ch_names
                if new_ch_names == old_ch_names:
                    raw = raw
                else:
                    mapping = {k: v for k, v in zip(old_ch_names, new_ch_names)}
                    raw.rename_channels(mapping)

                #raw = raw.drop_channels(['T7', 'T8', 'O1', 'Fp1', 'Fp2'], on_missing = 'ignore')

                # convert the V-EOG and H-EOG channels from EEG to EOG channels.
                raw = raw.set_channel_types({'V-EOG': 'eog', 'H-EOG': 'eog'}, verbose=None)

                # Set the montage of the channels to 10-20 international system.
                # Load the montage
                montage = mne.channels.make_standard_montage('standard_1020' )

                # Set the montage for the EEG data
                raw = raw.set_montage(montage)


                ## reapairing artifact with regression
                # Perform regression using the EOG channels/sensors as independent variable and the EEG
                # channels/sensors as dependent variables
                model_plain = EOGRegression(picks='eeg', exclude='bads', picks_artifact='eog', proj=True).fit(raw)
                raw_clean = model_plain.apply(raw)

                 # determine the number of independent components, choose method algorithm
                ica = ICA(n_components=None, method='picard',fit_params=dict(extended=True),
                                      random_state=21, max_iter = 10000 )


                # Apply ICA 
                # fit ICA with the data
                ica = ica.fit(raw_clean)
                # find components of ICA that contains artifact 
                bad_idxV, scoresV = ica.find_bads_eog(raw_clean, ch_name='V-EOG', threshold=2, start=None,
                                            stop=None, l_freq=0.5, h_freq=60,reject_by_annotation=True, 
                                                    measure='zscore', verbose=None)

                bad_idxH, scoresH = ica.find_bads_eog(raw_clean, ch_name='H-EOG', threshold=2, start=None,
                                            stop=None, l_freq=0.5, h_freq=60,reject_by_annotation=True, 
                                                    measure='zscore', verbose=None)
     
                # remove the components that violates threshold. 
                exclude = bad_idxV + bad_idxH
                ica.exclude = exclude
                raw_clean = ica.apply(raw_clean)

                # Find events from annotation
                events = mne.events_from_annotations(raw_clean)
                events = events[0]
                event_ids = {'standard/stimulus': 1, 'reaction_time/stimulus': 2,'distractor/stimulus': 3, 
                             'target/stimulus': 4}

                # epoch the data based on the onset of the events 
                tmin = -0.1
                tmax = 1.5
                epochs = mne.Epochs(raw_clean, events, event_id = event_ids, tmin = tmin, tmax = tmax, 
                                    preload = True, on_missing='ignore', baseline = None)

                baseline = (None, 0)
                # Apply baseline correction to the epochs
                epochs.apply_baseline(baseline=baseline)

                # Drop epochs based on the reject criteria
                reject = dict(eeg=120e-6) 
                epochs = epochs.drop_bad(reject=reject)
                # 
                epochs = epochs.drop_channels(['V-EOG','H-EOG','T7', 'T8', 'O1', 'Fp1', 'Fp2'], on_missing = 'ignore')
                
                # resample the data to 128 Hz to save time and memory
                epochs = epochs.resample(sfreq=128)
                dataset.append(epochs)
   
        return dataset

        #################### Apply Wavelet ######################################
    def applyWavelet(self, dataset):
        """
        This function takes an ndarray of shape (epochs, channel, sample) as input 
        and denoise it using wavelet denoising techniques.
        
        
        parameter:
                dataset: dataset of type ndarray with shape ( epochs, channel, sample)
                
        return: 
                ndarray with same shape as input
       
        """
        # list for ndarray of shape (epochs, channel, sample)
        cleaned_dataset = []
        for channels in dataset:
            # list for nd array of shape (channel, sample)
            denoised_channels = []
            for epochs in channels:
                #
                clean = denoise_wavelet(epochs, method = 'BayesShrink', mode = 'soft', 
                                    wavelet_levels = 2, wavelet = 'coif5', rescale_sigma = True)
                denoised_channels.append(clean)
            # reshape the cleaned channels as the input dataset    
            cleaned_dataset.append(np.array(denoised_channels).reshape(np.array(channels).shape ))
        return np.array(cleaned_dataset)

    
    def applyNotch(self, dataset):
        notch_filtered = mne.filter.notch_filter(dataset, Fs = 1000, freqs = 60, 
                                        filter_length='auto', notch_widths=None, trans_bandwidth=1, 
                                        method='fir', iir_params=None, mt_bandwidth=None, p_value=0.05, 
                                        picks=None, n_jobs=None, copy=True, phase='zero', 
                                        fir_window='hamming', fir_design='firwin', pad='reflect_limited', 
                                                           verbose=None)
        return notch_filtered


    def getStimuli(self, dataset):
        """
        This function takes the dataset for one of the subject groups as input
        and can performs the following operations on the data:
                a. Extract sub-epochs from the main Epochs dataset.
                b. Apply wavelet denoising to the sub-epochs, if the apply_wavelet
                put to be True.
                c. Apply a Notch-Filter to the sub-epochs, if apply_notch put to 
                be True. 

        parameters:
                1. dataset: a list of ndarray with the shape (epochs, channels, samples)
        return:
                
                1. Standard:      list of ndarray containing sub-epochs 
                                  belonging to standard stimuli
                                  
                2. Target:        list of ndarray containing sub-epochs 
                                  belonging to target stimuli.
                                  
                3. Distractor:    list of ndarray containing sub-epochs
                                  belonging to Distractor stimuli.
                                  
                4. Reaction_time  list of ndarray containing sub-epochs 
                                  belonging to reaction time.
                                  
                5. All_stimuli    list of ndarray containing all the four 
                                  stimuli
        """
        all_stimulis = []
        standard = []
        target = []
        distractor = []
        rt = []
        for epochs in dataset:
            # extract the standard part of the data
            standard_data = epochs['standard']
            standard_data = standard_data[:300]
            standard_data = standard_data.get_data()
            
            target_data = epochs['target']
            target_data = target_data[:75]
            target_data = target_data.get_data()
            
            distractor_data = epochs['distractor']
            distractor_data = distractor_data[:75]
            distractor_data = distractor_data.get_data()
            
            rt_data = epochs['reaction_time']
            rt_data = rt_data[:75]
            rt_data = rt_data.get_data()
            
            #########################
            # apply wavelet denoising
            if self.apply_wavelet == True:
                # clean data belonging to standard stimuli
                if isinstance(standard_data, np.ndarray):
                    # get array data from mne data
                    standard_data = standard_data
                    # clean the data with wavelet
                    standard_data = self.applyWavelet(standard_data)
                else:
                    standard_data = standard_data.get_data()
                    standard_data = self.applyWavelet(standard_data)

                
                # clean data belonging to target stimuli
                if isinstance(target_data, np.ndarray):
                    target_data = target_data
                    target_data = self.applyWavelet(target_data)
                else:
                    target_data = target_data.get_data()
                    target_data = self.applyWavelet(target_data)


                # clean data belonging to distractor stimuli
                if isinstance(distractor_data, np.ndarray): 
                    distractor_data = distractor_data
                    distractor_data = self.applyWavelet(distractor_data)
                else:
                    distractor_data = distractor_data.get_data()


                if isinstance(rt_data, np.ndarray):
                    rt_data = rt_data
                    rt_data = self.applyWavelet(rt_data)
                else:
                    rt_data = rt_data.get_data()
                    rt_data = self.applyWavelet(rt_data)

            else:
                pass
                 
            # Aplly nothch filtering
            if self.apply_notch == True:
                # clean data belonging to standard stimuli
                if isinstance(standard_data, np.ndarray):
                    # get array data from mne data
                    standard_data = standard_data
                    # clean the data with wavelet
                    standard_data = self.applyNotch(standard_data)
                else:
                    standard_data = standard_data.get_data()
                    # clean the data with wavelet
                    standard_data = self.applyNotch(standard_data)

                # clean data belonging to target stimuli
                if isinstance(target_data, np.ndarray):
                # get array data from mne data
                    target_data = target_data
                    target_data = self.applyNotch(target_data)
                else:
                    target_data = target_data.get_data()
                    target_data = self.applyNotch(target_data)


                # clean data belonging to distractor stimuli
                if isinstance(distractor_data, np.ndarray):
                    distractor_data = distractor_data
                    distractor_data = self.applyNotch(distractor_data)
                else:
                    distractor_data = distractor_data.get_data()
                    distractor_data = self.applyNotch(distractor_data)
              
                if isinstance(rt_data, np.ndarray):
                    rt_data = rt_data
                    rt_data = self.applyNotch(rt_data)
                else:
                    rt_data = rt_data.get_data()
                    rt_data = self.applyNotch(rt_data)
         
            else:
                pass
            
            if len(standard_data == 300):
                standard.append(standard_data)
            else:
                pass 
            if len(target_data == 75):
                target.append(target_data)
            else:
                pass
        
            if len(distractor_data== 75): 
                distractor.append(distractor_data)
            else:
                pass 
            
            rt.append(rt_data)
                    
            data = np.concatenate((standard_data, distractor_data, target_data, rt_data), 
                                  axis=0, out=None, dtype=None, casting="same_kind")
            all_stimulis.append(data)
        return {'Standard': standard, 'Target': target, 'Distractor': distractor, 
                'Reaction_time': rt, 'All_stimuli': all_stimulis}
    


# In[3]:


class readEpochDenoise:
    def __init__(self, DLB, AD, PDD, PD, HC, apply_wavelet = False, apply_notch = False):
        """
        In this class, the "Preprocessing" class is utilized to perform a series of
        steps on the data, including reading, cleaning, repairing, segmenting, and
        extracting epochs of oddball paradigms. The process involves the following steps:

            a. The data for the five subject groups is provided to the class. Each
            subject group's data is then fed one by one into the "Preprocessing" class
            for preprocessing. Within the "Preprocessing" class, the data for each
            subject group undergoes various preprocessing steps, such as cleaning,
            repairing artifacts, segmenting the data, and extracting epochs specific
            to the oddball stimuli.

            b. The preprocessed data is then saved into a file as a numpy array. This
            allows for the preservation of the preprocessed data, eliminating the need
            to rerun the preprocessing steps every time the analysis is performed
            
            

        
        parameters:
            DLB:  list of a path-like, ndarray or mne-format files (data for DLB subject group)
            AD:   list of a path-like, ndarray or mne-format files (data for AD subject group)  
            PDD:  list of a path-like, ndarray or mne-format files (data for PDD subject group)
            PD:   list of a path-like, ndarray or mne-format files (data for PD subject group)
            HC:   list of a path-like, ndarray or mne-format files (data for HC subject group)
            
            
            apply_wavelet: False|True
                If True: The data will be denoised by wavelet denoising
                If False: The data will not be denoised by wavlet
                
            apply_notch: False|True
                If True: The data will be filtered by nothch filtering technique.
                If False: The data will not be filtered by nothch filtering technique.

    
        """
        
        self.DLB = DLB
        self.AD = AD
        self.PDD = PDD
        self.PD = PD
        self.HC = HC
        
        self.apply_wavelet = apply_wavelet
        self.apply_notch = apply_notch
   
        
        self.DLB_data = preprocessing(path = self.DLB, 
                                    apply_wavelet = self.apply_wavelet,
                                     apply_notch = self.apply_notch )

        
        self.AD_data = preprocessing(path = self.AD,
                                     apply_wavelet = self.apply_wavelet,
                                     apply_notch = self.apply_notch)

        
        
        self.PDD_data = preprocessing(path = self.PDD,
                                    apply_wavelet = self.apply_wavelet,
                                     apply_notch = self.apply_notch )

        
        
        self.PD_data = preprocessing(path = self.PD,
                                     apply_wavelet = self.apply_wavelet,
                                    apply_notch = self.apply_notch)
        
        self.HC_data = preprocessing(path = self.HC,
                                      apply_wavelet = self.apply_wavelet,
                                     apply_notch = self.apply_notch)
    
    
    def save_data(self, standard = False, target = False, distractor = False, 
                     reaction_time = False, all_stimuli = False):
        
        """
        This function saves the preprocessed EEG data into a file as a ndarray
        using the following steps:
            a. First, the class ("readEpochDenoise") is fed with data for all
            the subject groups and executed.
            b. When calling this function, one of the input parameters/data is 
            chosen to be True. This step is repeated for all the subject groups,
            ensuring that the desired data is selected and saved into the file
            as an ndarray.
            
        By following these steps, the preprocessed EEG data for each subject grou
        p can be saved separately, allowing for easy access and utilization in 
        subsequent analyses.
        
        parameters: 
            standard:      False|True
            target:        False|True
            distractor:    False|True
            reaction_time: False|True
            all_stimuli:   False|True

        """
        
        if standard == True:
            events = 'Standard'
        elif target == True:
            events = 'Target'
        elif distractor == True:
            events = 'Distractor'
        elif reaction_time == True:
            events = 'Reaction_time'
        elif all_stimuli == True:
            events = 'All_stimuli'
        
        DLB_data = self.DLB_data.stimuli.get(events)
        
        AD_data  = self.AD_data.stimuli.get(events)
        
        PDD_data = self.PDD_data.stimuli.get(events)
  
        PD_data  = self.PD_data.stimuli.get(events)
    
        HC_data  = self.HC_data.stimuli.get(events)
        
        subject_list = [DLB_data, AD_data, PDD_data, PD_data, HC_data]
        
        groups = ['DLB', 'AD', 'PDD', 'PD', 'HC']
        
        for i, j in zip(groups, subject_list):
            path = os.path.join(f'preprocessed_data/{events}')
            file_name = path + '_' + '{0}.npy'.format(i)
            np.save(os.path.join(file_name), j)

