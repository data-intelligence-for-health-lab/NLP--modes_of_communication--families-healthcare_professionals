###############################################################################
#                                                                             #
#                        machine learning approach part 2                     #
#                                neural networks                              #
#                                                                June 23 2020 #
###############################################################################


### Loading libraries #########################################################
import time
import numpy as np
seed = np.random.seed(1)
import pandas as pd
pd.options.mode.chained_assignment = None
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras import backend as K
from keras.backend import clear_session
from sklearn.metrics import recall_score, confusion_matrix, roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from scipy import stats
import math 
import pickle
######################################################## Loading libraries ####


### Declaring I/O variables ###################################################
input_file = 'pre-processed_data.pickle'
output_file = 'ML_summary_part2.pickle'
################################################## Declaring I/O variables ####


### Declaring Functions #######################################################
def specifitiy(y, y_pred):
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    return (tn / (tn + fp))
###################################################### Declaring Functions ####


### Main routine ##############################################################
# Registering initial time
a = time.time()    
print("--start--")

# Open input file
datasets = pd.read_pickle(input_file)

k = 10

columns = ['n', 'DB', 'Level', 'Column',
           'n_0', 'n_1',
           'Sensitivity Train (95% CI)', 'Specificity Train (95% CI)', 'AUC Train (95% CI)',
           'Sensitivity Validation (95% CI)', 'Specificity Validation (95% CI)', 'AUC Validation (95% CI)',
           'Sensitivity Test', 'Specificity Test', 'AUC Test',
           'Best_Classifier', 'Best_Parameters'
           ]

output_summary = pd.DataFrame(columns = columns)

n_datasets = len(datasets['info'])

ngram_ranges = [(1,1), (1,2), (1,3)]
max_dfs = [0.7, 0.8, 0.9, 0.95, 1.0]
min_dfs = [2, 10, 50]
binarys = [False, True]
use_idfs = [False, True]
norms = ['l1', 'l2', None]
optimizers = ['adam']

n_combinations = len(ngram_ranges) * len(max_dfs) * len(min_dfs) * \
                  len(binarys) * len(use_idfs) * len(norms) * \
                  len(optimizers) 

for n in range(1, 157):   
    print()
    print('Processing dataset number: ',n)
    validation_scores = pd.DataFrame(columns = ['n',
                                                'ngram_range',
                                                'max_df',
                                                'min_df',
                                                'binary',
                                                'use_idf',
                                                'norm',
                                                'optimizer'])
    
    # Loading dataset info
    dataset_info = datasets['info'].loc[n,:]
    n_0 = dataset_info['n_0']
    n_1 = dataset_info['n_1']
    db_info = dataset_info['data_option']
    level_info = dataset_info['level']
    column_info = dataset_info['column']
    go_on = dataset_info['go_on']                        

    if go_on == True:  
        combination_summary = pd.DataFrame()
        dataset = datasets[n]
        X_train_validation = dataset['X_train_validation']
        y_train_validation = dataset['y_train_validation']        
        X_test = dataset['X_test']
        y_test = dataset['y_test']
        
        vectorizer_dict = {}
        
        combination = 0
        AUC_mean_validation = '0.000'
        for ngram_range in ngram_ranges:
            if AUC_mean_validation == '1.000':
                break
            for max_df in max_dfs:
                if AUC_mean_validation == '1.000':
                    print('AUC_mean_validation == 1.000')
                    break
                for min_df in min_dfs:
                    if AUC_mean_validation == '1.000':
                        print('AUC_mean_validation == 1.000')
                        break
                    for binary in binarys:
                        if AUC_mean_validation == '1.000':
                            print('AUC_mean_validation == 1.000')
                            break
                        for use_idf in use_idfs:
                            if AUC_mean_validation == '1.000':
                                print('AUC_mean_validation == 1.000')
                                break
                            for norm in norms: 
                                if AUC_mean_validation == '1.000':
                                    print('AUC_mean_validation == 1.000')
                                    break
                                for optimizer in optimizers:
                                    if AUC_mean_validation == '1.000':
                                        print('AUC_mean_validation == 1.000')
                                        break
                                             
                                    a11 = time.time()
                                    kfold = StratifiedKFold(n_splits = k, shuffle = True, random_state = seed)
                                    
                                    sensitivity_train_list = []
                                    specificity_train_list  = [] 
                                    auc_train_list  = []                      
                                    sensitivity_validation_list  = []
                                    specificity_validation_list  = []
                                    auc_validation_list  = []                                        
                              
                                    fold = 0
                                    for train_index, validation_index in kfold.split(X_train_validation, y_train_validation):  
                                        a1 = time.time() 
                                        X_train, y_train, X_validation, y_validation = X_train_validation.iloc[train_index], y_train_validation.iloc[train_index], X_train_validation.iloc[validation_index], y_train_validation.iloc[validation_index]
   
                                        print()
                                        print('Processing dataset number: ',n)
                                        print('combination: ', combination, 'out of: ', n_combinations)
                                        print('ngram_range: ',ngram_range)
                                        print('max_df: ',max_df)
                                        print('min_df: ',min_df)
                                        print('binary: ',binary)
                                        print('use_idf: ',use_idf)
                                        print('norm: ',norm)
                                        print('optimizer: ',optimizer)
                                        print('Fold: ',fold)
                                        print()
                                        
                                        vectorizer = TfidfVectorizer(
                                                ngram_range = ngram_range,
                                                max_df = max_df,
                                                min_df = min_df,
                                                binary = binary,
                                                use_idf = use_idf,
                                                norm = norm, 
                                                )
                                        
                                        X_train = vectorizer.fit_transform(X_train)
                                        X_validation = vectorizer.transform(X_validation)
                                        
                                        X_train = X_train.todense()
                                        X_validation = X_validation.todense()
                                        
                                        y_train = y_train.to_numpy()
                                        y_validation = y_validation.to_numpy()
                                        
                                        n_feat = X_train.shape[1]
                                        if n_feat > 2048:
                                            n_feat = 2048                                            
                                        
                                        model = Sequential()
                                        model.add(Dense(n_feat,activation='relu'))
                                        model.add(Dense(1,activation='sigmoid',))
                                        model.compile(optimizer = optimizer, 
                                                      loss = 'binary_crossentropy', 
                                                      metrics = ['binary_accuracy'])
                                                                                    
                                        model.fit(X_train, 
                                                  y_train, 
                                                  epochs = 1000, 
                                                  validation_data = (X_validation, y_validation), 
                                                  verbose = 0,
                                                  shuffle = False, 
                                                  initial_epoch = 0,
                                                  callbacks=[EarlyStopping(monitor='val_loss', min_delta = 0.01)]
                                                  )
                                        
                                        y_pred_train = model.predict(X_train)
                                        y_pred_validation = model.predict(X_validation)
                                        
                                        clear_session()
                                        
                                        # Calculating perfomance metrics
                                        sensitivity_train_fold_list = []
                                        specificity_train_fold_list  = [] 
                                        auc_train_fold_list  = []                      
                                        sensitivity_validation_fold_list  = []
                                        specificity_validation_fold_list  = []
                                        auc_validation_fold_list  = [] 
                                        threshold_index = []
                                        
                                        for threshold in np.arange(0.01,1,0.01):
                                            threshold_index.append(threshold)
                                            y_pred_train_temp = [1 if prediction >= threshold else 0 for prediction in y_pred_train]
                                            y_pred_validation_temp = [1 if prediction >= threshold else 0 for prediction in y_pred_validation]
                                            
                                            sensitivity_train = recall_score(y_train, y_pred_train_temp)
                                            specificity_train = specifitiy(y_train, y_pred_train_temp)
                                            auc_train = roc_auc_score(y_train, y_pred_train_temp)                                                
                                            sensitivity_validation = recall_score(y_validation, y_pred_validation_temp)
                                            specificity_validation = specifitiy(y_validation, y_pred_validation_temp)
                                            auc_validation = roc_auc_score(y_validation, y_pred_validation_temp)
                                        
                                            sensitivity_train_fold_list.append(sensitivity_train)
                                            specificity_train_fold_list.append(specificity_train)
                                            auc_train_fold_list.append(auc_train)
                                            sensitivity_validation_fold_list.append(sensitivity_validation)
                                            specificity_validation_fold_list.append(specificity_validation)
                                            auc_validation_fold_list.append(auc_validation)

                                        sensitivity_train_list.append(sensitivity_train_fold_list)
                                        specificity_train_list.append(specificity_train_fold_list)
                                        auc_train_list.append(auc_train_fold_list)
                                        sensitivity_validation_list.append(sensitivity_validation_fold_list)
                                        specificity_validation_list.append(specificity_validation_fold_list)
                                        auc_validation_list.append(auc_validation_fold_list)
                                        
                                        if fold == 0:
                                            vectorizer_dict[combination] = {fold : vectorizer}
                                        else:
                                            vectorizer_dict[combination].update({fold : vectorizer})
                                        fold += 1
                                        
                                        b1 = time.time() 
                                        print('Fold processing time: %0.2f minutos' %((b1-a1)/60))  
                                        print()
                                        
                                    auc_threshold = []
                                    auc_threshold_max_fold = []
                                    for threshold in range(0,99):
                                        auc_temp = []
                                        for f in range(0,fold):
                                            auc_temp.append(auc_validation_list[f][threshold])                                       
                                        # Identify the fold that had the best AUC for each threshold
                                        auc_threshold_max_fold.append(auc_temp.index(max(auc_temp)))
                                        auc_threshold.append(np.mean(auc_temp))
                                    best_threshold_n = auc_threshold.index(max(auc_threshold))
                                    best_threshold = threshold_index[best_threshold_n]
                                    reference_fold = auc_threshold_max_fold[best_threshold_n]
                                    best_threshold = np.round(best_threshold,3)
                                    
                                    sensitivity_train = []
                                    specificity_train = []
                                    AUC_train = []                                
                                    sensitivity_validation = []
                                    specificity_validation = []
                                    AUC_validation = []
                                    for f in range(0,fold):
                                        sensitivity_train.append(sensitivity_train_list[f][best_threshold_n])
                                        specificity_train.append(specificity_train_list[f][best_threshold_n])
                                        AUC_train.append(auc_train_list[f][best_threshold_n])       
                                        sensitivity_validation.append(sensitivity_validation_list[f][best_threshold_n])
                                        specificity_validation.append(specificity_validation_list[f][best_threshold_n])
                                        AUC_validation.append(auc_validation_list[f][best_threshold_n])
                        
                                    # sensitivity train              
                                    sensitivity_mean_train = '{:1.3f}'.format(round(np.mean(sensitivity_train), 3))
                                    sensitivity_LB_train = np.mean(sensitivity_train) - stats.t.ppf(1-0.025, k - 1)*np.std(sensitivity_train)/math.sqrt(k)
                                    if sensitivity_LB_train < 0:
                                        sensitivity_LB_train = 0                                    
                                    sensitivity_UB_train = np.mean(sensitivity_train) + stats.t.ppf(1-0.025, k - 1)*np.std(sensitivity_train)/math.sqrt(k)
                                    if sensitivity_UB_train > 1:
                                        sensitivity_UB_train = 1
                                    sensitivity_LB_train = '{:1.3f}'.format(sensitivity_LB_train,3)
                                    sensitivity_UB_train = '{:1.3f}'.format(sensitivity_UB_train,3)
                                    
                                    # sensitivity validation 
                                    sensitivity_mean_validation = '{:1.3f}'.format(round(np.mean(sensitivity_validation), 3))
                                    sensitivity_LB_validation = np.mean(sensitivity_validation) - stats.t.ppf(1-0.025, k - 1)*np.std(sensitivity_validation)/math.sqrt(k)
                                    if sensitivity_LB_validation < 0:
                                        sensitivity_LB_validation = 0
                                    sensitivity_UB_validation = np.mean(sensitivity_validation) + stats.t.ppf(1-0.025, k - 1)*np.std(sensitivity_validation)/math.sqrt(k)
                                    if sensitivity_UB_validation > 1:
                                        sensitivity_UB_validation = 1
                                    sensitivity_LB_validation = '{:1.3f}'.format(sensitivity_LB_validation,3)   
                                    sensitivity_UB_validation = '{:1.3f}'.format(sensitivity_UB_validation,3) 
                        
                                    # Specificity train
                                    specificity_mean_train = '{:1.3f}'.format(round(np.mean(specificity_train), 3))
                                    specificity_LB_train = np.mean(specificity_train) - stats.t.ppf(1-0.025, k - 1)*np.std(specificity_train)/math.sqrt(k)
                                    if specificity_LB_train < 0:
                                        specificity_LB_train = 0
                                    specificity_UB_train = np.mean(specificity_train) + stats.t.ppf(1-0.025, k - 1)*np.std(specificity_train)/math.sqrt(k)
                                    if specificity_UB_train > 1:
                                        specificity_UB_train = 1
                                    specificity_LB_train = '{:1.3f}'.format(specificity_LB_train,3)
                                    specificity_UB_train = '{:1.3f}'.format(specificity_UB_train,3)
                                    
                                    # Specificity validation
                                    specificity_mean_validation = '{:1.3f}'.format(round(np.mean(specificity_validation), 3))
                                    specificity_LB_validation = np.mean(specificity_validation) - stats.t.ppf(1-0.025, k - 1)*np.std(specificity_validation)/math.sqrt(k)
                                    if specificity_LB_validation < 0:
                                        specificity_LB_validation = 0
                                    specificity_UB_validation = np.mean(specificity_validation) + stats.t.ppf(1-0.025, k - 1)*np.std(specificity_validation)/math.sqrt(k)
                                    if specificity_UB_validation > 1:
                                        specificity_UB_validation = 1
                                    specificity_LB_validation = '{:1.3f}'.format(specificity_LB_validation,3)   
                                    specificity_UB_validation = '{:1.3f}'.format(specificity_UB_validation,3)  
                                    
                                    # AUC train
                                    AUC_mean_train = '{:1.3f}'.format(round(np.mean(AUC_train), 3))
                                    AUC_LB_train = np.mean(AUC_train) - stats.t.ppf(1-0.025, k - 1)*np.std(AUC_train)/math.sqrt(k)
                                    if AUC_LB_train < 0:
                                        AUC_LB_train = 0
                                    AUC_UB_train = np.mean(AUC_train) + stats.t.ppf(1-0.025, k - 1)*np.std(AUC_train)/math.sqrt(k)
                                    if AUC_UB_train > 1:
                                        AUC_UB_train = 1
                                    AUC_LB_train = '{:1.3f}'.format(AUC_LB_train,3)
                                    AUC_UB_train = '{:1.3f}'.format(AUC_UB_train,3)
                                    
                                    # AUC validation
                                    AUC_mean_validation = '{:1.3f}'.format(round(np.mean(AUC_validation), 3))
                                    AUC_LB_validation = np.mean(AUC_validation) - stats.t.ppf(1-0.025, k - 1)*np.std(AUC_validation)/math.sqrt(k)
                                    if AUC_LB_validation < 0:
                                        AUC_LB_validation = 0
                                    AUC_UB_validation = np.mean(AUC_validation) + stats.t.ppf(1-0.025, k - 1)*np.std(AUC_validation)/math.sqrt(k)
                                    if AUC_UB_validation > 1:
                                        AUC_UB_validation = 1
                                    AUC_LB_validation = '{:1.3f}'.format(AUC_LB_validation,3)    
                                    AUC_UB_validation = '{:1.3f}'.format(AUC_UB_validation,3) 

                                    # formating metrics for output
                                    sensitivity_train = sensitivity_mean_train+' ('+sensitivity_LB_train+'-'+sensitivity_UB_train+')'
                                    specificity_train = specificity_mean_train+' ('+specificity_LB_train+'-'+specificity_UB_train+')'
                                    AUC_train = AUC_mean_train+' ('+AUC_LB_train+'-'+AUC_UB_train+')'
                                    sensitivity_validation = sensitivity_mean_validation+' ('+sensitivity_LB_validation+'-'+sensitivity_UB_validation+')'
                                    specificity_validation = specificity_mean_validation+' ('+specificity_LB_validation+'-'+specificity_UB_validation+')'
                                    AUC_validation = AUC_mean_validation+' ('+AUC_LB_validation+'-'+AUC_UB_validation+')'
                                    
                                    parameters = ', '.join(['ngram_range: '+str(ngram_range)] + 
                                                            ['max_df: '+str(max_df)] + 
                                                            ['min_df: '+str(min_df)] + 
                                                            ['binary: '+str(binary)] + 
                                                            ['use_idf: '+str(use_idf)] + 
                                                            ['norm: '+str(norm)] + 
                                                            ['optimizer: '+str(optimizer)])

                                    # saving info of this round
                                    combination_summary.loc[combination,'combination'] = combination
                                    combination_summary.loc[combination,'ngram_range'] = str(ngram_range)
                                    combination_summary.loc[combination,'max_df'] = str(max_df)
                                    combination_summary.loc[combination,'min_df'] = str(min_df)
                                    combination_summary.loc[combination,'binary'] = str(binary)
                                    combination_summary.loc[combination,'use_idf'] = str(use_idf)
                                    combination_summary.loc[combination,'norm'] = str(norm)
                                    combination_summary.loc[combination,'optimizer'] = str(optimizer)
                                    combination_summary.loc[combination,'Threshold'] = best_threshold         
                                    combination_summary.loc[combination,'reference_fold'] = reference_fold                                      
                                    combination_summary.loc[combination,'Sensitivity Train (95% CI)'] = sensitivity_train
                                    combination_summary.loc[combination,'Specificity Train (95% CI)'] = specificity_train
                                    combination_summary.loc[combination,'AUC Train (95% CI)'] = AUC_train
                                    combination_summary.loc[combination,'Sensitivity Validation (95% CI)'] = sensitivity_validation
                                    combination_summary.loc[combination,'Specificity Validation (95% CI)'] = specificity_validation
                                    combination_summary.loc[combination,'AUC Validation (95% CI)'] = AUC_validation
                                    combination += 1
                                    
                                    b11 = time.time() 
                                    print('AUC Validation (95% CI): ', AUC_validation)
                                    print('Combination processing time: %0.2f minutos' %((b11-a11)/60))  
                                    print()                                        

        combination_summary = combination_summary.sort_values(by = 'AUC Validation (95% CI)', ascending = False).reset_index(drop = True)
        best_combination = combination_summary.loc[0, 'combination']
        best_ngram_range = combination_summary.loc[0, 'ngram_range']
        best_max_df = combination_summary.loc[0, 'max_df']
        best_min_df = combination_summary.loc[0, 'min_df']
        best_binary = combination_summary.loc[0, 'binary'] 
        best_use_idf = combination_summary.loc[0, 'use_idf']
        best_norm = combination_summary.loc[0, 'norm']        
        best_optimizer = combination_summary.loc[0, 'optimizer']
        best_threshold = combination_summary.loc[0, 'Threshold']   
        best_reference_fold = combination_summary.loc[0, 'reference_fold']                                            
        best_sensitivity_train = combination_summary.loc[0,'Sensitivity Train (95% CI)'] 
        best_specificity_train = combination_summary.loc[0,'Specificity Train (95% CI)']
        best_AUC_train = combination_summary.loc[0,'AUC Train (95% CI)']
        best_sensitivity_validation = combination_summary.loc[0,'Sensitivity Validation (95% CI)']
        best_specificity_validation = combination_summary.loc[0,'Specificity Validation (95% CI)']
        best_AUC_validation = combination_summary.loc[0,'AUC Validation (95% CI)']
        best_param = ', '.join(
                ['ngram_range: '+str(best_ngram_range)] + 
                ['max_df: '+str(best_max_df)] + 
                ['min_df: '+str(best_min_df)] + 
                ['binary: '+str(best_binary)] + 
                ['use_idf: '+str(best_use_idf)] + 
                ['norm: '+str(best_norm)] + 
                ['optimizer: '+str(best_optimizer)] + 
                ['Threshold: '+str(best_threshold)])
    
        vectorizer = vectorizer_dict[best_combination][best_reference_fold]                                            
                                            
        X_train_validation = vectorizer.transform(X_train_validation)
        X_test = vectorizer.transform(X_test)
        
        X_train_validation = X_train_validation.todense()
        X_test = X_test.todense()
        
        y_train_validation = y_train_validation.to_numpy()
        y_test = y_test.to_numpy()

        n_feat = X_train_validation.shape[1]
        if n_feat > 2048:
            n_feat = 2048                                            
        
        model = Sequential()
        model.add(Dense(n_feat,activation='relu'))
        model.add(Dense(1,activation='sigmoid',))
        model.compile(optimizer = optimizer, 
                      loss = 'binary_crossentropy', 
                      metrics = ['binary_accuracy'])
                
                                           
        model.fit(X_train_validation, 
                  y_train_validation, 
                  epochs = 1000, 
                  validation_data = None, 
                  verbose = 0,
                  shuffle = False, 
                  initial_epoch = 0,
                  callbacks=[EarlyStopping(monitor='loss', min_delta = 0.01)]
                  )

        y_pred_test = model.predict(X_test)
        y_pred_test = [1 if prediction >= best_threshold else 0 for prediction in y_pred_test]

        # evaluating performance of test set        
        sensitivity_test = '{:1.3f}'.format(round(recall_score(y_test, y_pred_test), 3))
        specificity_test = '{:1.3f}'.format(round(specifitiy(y_test, y_pred_test), 3))
        auc_test = '{:1.3f}'.format(round(roc_auc_score(y_test, y_pred_test), 3))
        
        clear_session()

        # Registering results
        output_summary.loc[n,'n'] = n
        output_summary.loc[n,'DB'] = db_info
        output_summary.loc[n,'Level'] = level_info
        output_summary.loc[n,'Column'] = column_info
        output_summary.loc[n,'n_0'] = n_0
        output_summary.loc[n,'n_1'] = n_1
        output_summary.loc[n,'Sensitivity Train (95% CI)'] = best_sensitivity_train
        output_summary.loc[n,'Specificity Train (95% CI)'] = best_specificity_train
        output_summary.loc[n,'AUC Train (95% CI)'] = best_AUC_train
        output_summary.loc[n,'Sensitivity Validation (95% CI)'] = best_sensitivity_validation
        output_summary.loc[n,'Specificity Validation (95% CI)'] = best_specificity_validation
        output_summary.loc[n,'AUC Validation (95% CI)'] = best_AUC_validation
        output_summary.loc[n,'Sensitivity Test'] = sensitivity_test
        output_summary.loc[n,'Specificity Test'] = specificity_test
        output_summary.loc[n,'AUC Test'] = auc_test
        output_summary.loc[n,'Best_Classifier'] = 'Neural Network'
        output_summary.loc[n,'Best_Parameters'] = best_param
    else:
        output_summary.loc[n,'n'] = n
        output_summary.loc[n,'DB'] = db_info
        output_summary.loc[n,'Level'] = level_info
        output_summary.loc[n,'Column'] = column_info
        output_summary.loc[n,'n_0'] = n_0
        output_summary.loc[n,'n_1'] = n_1
        output_summary.loc[n,'Sensitivity Train (95% CI)'] = 'N/A'
        output_summary.loc[n,'Specificity Train (95% CI)'] = 'N/A'
        output_summary.loc[n,'AUC Train (95% CI)'] = 'N/A'
        output_summary.loc[n,'Sensitivity Validation (95% CI)'] = 'N/A'
        output_summary.loc[n,'Specificity Validation (95% CI)'] = 'N/A'
        output_summary.loc[n,'AUC Validation (95% CI)'] = 'N/A'
        output_summary.loc[n,'Sensitivity Test'] = 'N/A'
        output_summary.loc[n,'Specificity Test'] = 'N/A'
        output_summary.loc[n,'AUC Test'] = 'N/A'
        output_summary.loc[n,'Best_Classifier'] = 'N/A'
        output_summary.loc[n,'Best_Parameters'] = 'N/A'
    
    with open(output_file, 'wb') as x:
        pickle.dump(output_summary, x, protocol=pickle.HIGHEST_PROTOCOL)     
   
    print()
    print()
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    print('          Saving results for dataset number: ', n,)
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    print()
    print()

# Registering final time
b = time.time()        
print('--end--')
print('Total processing time: %0.2f minutos' %((b-a)/60))    
############################################################# Main routine ####