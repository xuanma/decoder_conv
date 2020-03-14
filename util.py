import scipy.io as sio
import pickle
import numpy as np
import fnmatch, os

cnames = [
'#FF0000', #red   0
'#000000', #black   1
'#00FFFF', #aqua or cyan   2
'#A52A2A', #brown   3
'#FFFF00', #yellow   4
'#0000FF', #blue   5
'#008000', #green   6
'#800080', #purple   7
'#FF69B4', #hotpink   8
'#008B8B', #'darkcyan'   9
'#9400D3', #'darkviolet'   10 
'#FFA500', #'orange   11
'#808000', #'olive'   12
'#FF1493', #deeppink   13
'#FF7F50', #'coral'   14
'#8B0000', #darkred   15
'#808080'] #gray    16

def batch_load(base_path,
               file_list,
               bin_size,
               smooth_window,
               denoising,
               smooth,
               remove_silent = 1,
               clf = ['./', 'rf_clf_20200218.joblib']):
    all_spike, all_emg, all_timeframe = [], [], []

    for i in range(len(file_list)):
        with open ( base_path + file_list[i], 'rb' ) as fp:
             my_cage_data = pickle.load(fp)
             my_cage_data.pre_processing_summary()
        if denoising == 1:
            my_cage_data.clean_cortical_data_with_classifier(clf[0], clf[1])
        my_cage_data.bin_data(bin_size)
        if smooth == 1:
            my_cage_data.smooth_binned_spikes('half_gaussian', smooth_window)
        st, et, tt = np.asarray(my_cage_data.binned['spikes']).T, np.asarray(my_cage_data.binned['filtered_EMG']).T, np.asarray(my_cage_data.binned['timeframe'])
        st, et, tt = st.astype(np.float32), et.astype(np.float32), tt.astype(np.float32)
        # -------- Determine which channels should be removed -------- #
        if i == 0:
            fr = []
            for n in range(len(my_cage_data.spikes)): 
                fr.append((len(my_cage_data.spikes[n])/my_cage_data.nev_duration))
            remove_idx = np.where(np.asarray(fr) < 1)[0]
        if remove_silent == 1:
           st = np.delete(st, remove_idx, axis = 1) 
        
        all_spike.append(st)
        all_emg.append(et)
        all_timeframe.append(tt)
        del(my_cage_data)
    return all_spike, all_emg, all_timeframe

def vaf(x,xhat):
    x = x - x.mean(axis=0)
    xhat = xhat - xhat.mean(axis=0)
    return (1-(np.sum(np.square(x - xhat))/np.sum(np.square(x))))    
    
def flatten_list(X):
    n_col = np.size(X[0],1)
    Y = np.empty((0, n_col))
    for each in X:
        Y = np.vstack((Y, each))
    return Y

def flatten_list_3d(X):
    n_c1 = np.size(X[0], 1)
    n_c2 = np.size(X[0], 2)
    Y = np.empty((0, n_c1, n_c2))
    for each in X:
        Y = np.vstack((Y, each))
    return Y

def fix_bad_array(dataset, bad_chs):
    """
    dataset: xds object
    bad_chs: a list containing the number part of electrode name, like the '9' in 'elec 9'
    """
    full_idx = np.arange(0, len(dataset.unit_names), 1)
    bad_idx = []
    for each in bad_chs:
        a = 'elec'+str(each)
        if a in dataset.unit_names:
            temp = dataset.unit_names.index(a)
            bad_idx.append(temp)
    good_idx = np.delete(full_idx, bad_idx)
    return good_idx

def find_EMG_idx(dataset, EMG_list):
    EMG_names = np.asarray(dataset.EMG_names)
    idx = []
    for each in EMG_list:
        idx.append(np.where(EMG_names == each)[0])
    return np.asarray(idx).reshape((len(idx), ))    









