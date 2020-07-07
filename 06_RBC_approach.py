###############################################################################
#                                                                             #
#                            rule based approach                              #
#                                                                             #
#                                                                 July 6 2020 #
###############################################################################


### Loading libraries #########################################################
import time
import numpy as np
import pandas as pd
from scipy import stats
import math 
import pickle
import re
from sklearn.metrics import make_scorer, recall_score, confusion_matrix, roc_auc_score
######################################################## Loading libraries ####


### Declaring I/O variables ###################################################
input_file = 'pre-processed_data.pickle'
output_file = 'RBC_results_summary.xlsx'
aux_data = 'auxiliar_data.xlsx'
names_pickle = "names_dict.pickle"
################################################## Declaring I/O variables ####


### Declaring Functions #######################################################
def specifitiy(y, y_pred):
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    return (tn / (tn + fp))

def mount_list(file, sheet):
    df = pd.read_excel(file, sheet_name = sheet) 
    ref_label = list(df)[0]
    df.index = df[ref_label].str.len()
    df = df.sort_index(ascending = False).reset_index(drop=True)
    return df[ref_label].tolist()

def predict(X, column):
    cat = column[:1]
    params = X['ParameterID'].split()
    text = X['X']
    
    cond1_flag = False
    cond2_flag = False
    
#--- Evaluating condition #1 --------------------------------------------------
    # Category - Documented family or friends 
    if cat == 'F':   
        #  4957: Family Quick View Summary
        # 17202: Contact Information Family
        if ('4957' in params) | ('17202' in params):
            cond1_flag = True
        else:
            cond1_flag = False
            
    # Category - Visits            
    elif cat == 'V':
        incl_flag = False
        excl_flag = False
        
        # Inclusion terms
        incl_terms = cat_incl[cat_incl['CODE'] == cat]['TERM'].to_list()
        incl_terms = sorted(incl_terms, key=len, reverse = True) 
        
        # Inclusion params
        # 4962 : OLD_Family Visit Comment
        # 12362 : Comment Family In
        # 12371 : Comment Family Conference
        # 20325 : MD Comment Family Conference
        # 22512 : OLD_MD Present During Family Conference
        # 22546 : Comment Family Out
        
        # Exclusion param
        # 12346 : Comment Family Phone Call
        
        # Evaluating incl_params:        
        if (('4962' in params) | ('12362' in params) | ('12371' in params) | \
            ('20325' in params) | ('22512' in params) | ('22546' in params)):
            incl_flag = True
        else:
            # Evaluating incl_terms
            for term in incl_terms:
                if bool(re.search(term, text)):
                    incl_flag = True
                    break
                else:
                    incl_flag = False   
                    
        # Evaluating excl_params:        
        if ('12346' in params):
            excl_flag = True
        else:
            excl_flag = False
        
        # Evaluating condition #1
        if (incl_flag == True) & (excl_flag == False):
            cond1_flag = True
        else:
            cond1_flag = False
            
             
    # Category - Phone Calls            
    elif cat == 'P':
        incl_flag = False
        excl_flag = False
        
        # Inclusion / exclusion terms
        incl_terms = cat_incl[cat_incl['CODE'] == cat]['TERM'].to_list()
        incl_terms = sorted(incl_terms, key=len, reverse = True) 
        
        excl_terms = cat_excl[cat_excl['CODE'] == cat]['TERM'].to_list()
        excl_terms = sorted(excl_terms, key=len, reverse = True) 
        
        # Inclusion param
        # 12346 : Comment Family Phone Call
        
        # Exclusion params
        # 4962 : OLD_Family Visit Comment
        # 12362 : Comment Family In
        # 12371 : Comment Family Conference
        # 20325 : MD Comment Family Conference
        # 22512 : OLD_MD Present During Family Conference
        # 22546 : Comment Family Out
        
        # Evaluating incl_params:        
        if ('12346' in params):
            incl_flag = True
        else:
            # Evaluating incl_terms
            for term in incl_terms:
                if bool(re.search(term, text)):
                    incl_flag = True
                    break
                else:
                    incl_flag = False   
                    
        # Evaluating excl_params:        
        if (('4962' in params) | ('12362' in params) | ('12371' in params) | \
            ('20325' in params) | ('22512' in params) | ('22546' in params)):
            excl_flag = True
        else:
            # Evaluating excl_terms
            for term in excl_terms:
                if bool(re.search(term, text)):
                    excl_flag = True
                    break
                else:
                    excl_flag = False       
                    
        # Evaluating condition #1
        if (incl_flag == True) & (excl_flag == False):
            cond1_flag = True
        else:
            cond1_flag = False
#-------------------------------------------------- Evaluating condition #1 ---
   
#--- Evaluating condition #2 --------------------------------------------------
    if cond1_flag == True:
        incl_flag = False
        excl_flag = False

        # Inclusion terms
        incl_terms = sub_cat_incl[sub_cat_incl['CODE'] == column]['TERM'].to_list()
        incl_terms = sorted(incl_terms, key=len, reverse = True) 
        
        # Special case #1. Not specified
        if (column == 'F22') | (column == 'V45') | (column == 'V46') | \
            (column == 'P43') | (column == 'P44'):
            names = pd.read_pickle(names_pickle).keys()
            relations = sub_cat_incl[sub_cat_incl['CODE'] == 'F1']['TERM'].to_list()
                                      
            # Evaluating incl_terms
            for name in names:
                if bool(re.search(name, text)):
                    incl_flag = True
                    break
                else:
                    incl_flag = False
                
            # Evaluating excl_terms
            for relation in relations:
                if bool(re.search(relation, text)):
                    excl_flag = True
                    break
                else:
                    excl_flag = False
                    
            # Evaluating condition #2
            if (incl_flag == True) & (excl_flag == False):
                cond2_flag = True
            else:
                cond2_flag = False
                
        # Special case #2. Not specified - male
        if (column == 'V47') | (column == 'V48') | (column == 'P45') | (column == 'P46'):
            names = pd.read_pickle(names_pickle)
            relations = sub_cat_incl[sub_cat_incl['CODE'] == 'F1']['TERM'].to_list()
                                      
            # Evaluating incl_terms
            for name in names.keys():
                if names[name] == 'M':
                    if bool(re.search(name, text)):
                        incl_flag = True
                        break
                    else:
                        incl_flag = False
                
            # Evaluating excl_terms
            for relation in relations:
                if bool(re.search(relation, text)):
                    excl_flag = True
                    break
                else:
                    excl_flag = False
                    
            # Evaluating condition #2
            if (incl_flag == True) & (excl_flag == False):
                cond2_flag = True
            else:
                cond2_flag = False       
                
        # Special case #3. Not specified - female
        if (column == 'V51') | (column == 'V52') | (column == 'P47') | (column == 'P48'):
            names = pd.read_pickle(names_pickle)
            relations = sub_cat_incl[sub_cat_incl['CODE'] == 'F1']['TERM'].to_list()
                                      
            # Evaluating incl_terms
            for name in names.keys():
                if names[name] == 'F':
                    if bool(re.search(name, text)):
                        incl_flag = True
                        break
                    else:
                        incl_flag = False
                
            # Evaluating excl_terms
            for relation in relations:
                if bool(re.search(relation, text)):
                    excl_flag = True
                    break
                else:
                    excl_flag = False
                    
            # Evaluating condition #2
            if (incl_flag == True) & (excl_flag == False):
                cond2_flag = True
            else:
                cond2_flag = False   
                
        # Special case #4. Not specified - unknown
        if (column == 'V49') | (column == 'V50') | (column == 'P49') | (column == 'P50'):
            names = pd.read_pickle(names_pickle)
            relations = sub_cat_incl[sub_cat_incl['CODE'] == 'F1']['TERM'].to_list()
                                      
            # Evaluating incl_terms
            for name in names.keys():
                if names[name] == 'U':
                    if bool(re.search(name, text)):
                        incl_flag = True
                        break
                    else:
                        incl_flag = False
                
            # Evaluating excl_terms
            for relation in relations:
                if bool(re.search(relation, text)):
                    excl_flag = True
                    break
                else:
                    excl_flag = False
                    
            # Evaluating condition #2
            if (incl_flag == True) & (excl_flag == False):
                cond2_flag = True
            else:
                cond2_flag = False  
                
        # Special case #5. family meeting / conference related
        if ((column == 'V55') | (column == 'V56') | (column == 'V65') | \
           (column == 'V66') | (column == 'V67') | (column == 'V68') | \
           (column == 'V69') | (column == 'V70')): 
               
            # General inclusion param
            # 20325 : MD Comment Family Conference
            # 12371 : Comment Family Conference
            # 22512 : OLD_MD Present During Family Conference
                           
            # Evaluating general incl_params (V55 and V56):        
            if (('20325' in params) | ('12371' in params) | ('22512' in params)):
                incl_flag = True
            else:
                # Evaluating general incl_terms
                incl_terms = sub_cat_incl[sub_cat_incl['CODE'] == 'V55']['TERM'].to_list()
                incl_terms = sorted(incl_terms, key=len, reverse = True)                
                for term in incl_terms:
                    if bool(re.search(term, text)):
                        incl_flag = True
                        break
                    else:
                        incl_flag = False   
            
            # Is this a family meeting / conference?
            if ((column == 'V55') | (column == 'V56')):                
                # Evaluating condition #2
                if (incl_flag == True):
                    cond2_flag = True
                else:
                    cond2_flag = False  
                    
            # Is family meeting at beside?
            if ((column == 'V65') | (column == 'V66')):
                if incl_flag == True:
                    # Evaluating incl_terms
                    incl_terms = sub_cat_incl[sub_cat_incl['CODE'] == column]['TERM'].to_list()
                    incl_terms = sorted(incl_terms, key=len, reverse = True)                
                    for term in incl_terms:
                        if bool(re.search(term, text)):
                            incl_flag = True
                            break
                        else:
                            incl_flag = False   
                else:
                    incl_flag = False
                    
                # Evaluating condition #2
                if (incl_flag == True):
                    cond2_flag = True
                else:
                    cond2_flag = False                     
                    
            # Is family meeting at the conference room?
            if ((column == 'V67') | (column == 'V68')):
                if incl_flag == True:
                    # Evaluating general incl_terms
                    incl_terms = sub_cat_incl[sub_cat_incl['CODE'] == column]['TERM'].to_list()
                    incl_terms = sorted(incl_terms, key=len, reverse = True)                
                    for term in incl_terms:
                        if bool(re.search(term, text)):
                            incl_flag = True
                            break
                        else:
                            incl_flag = False   
                else:
                    incl_flag = False   
                    
                # Evaluating condition #2
                if (incl_flag == True):
                    cond2_flag = True
                else:
                    cond2_flag = False                     
                    
            # Is family meeting unspecified?
            if ((column == 'V69') | (column == 'V70')):
                excl_terms = sub_cat_excl[sub_cat_excl['CODE'] == column]['TERM'].to_list()
                excl_terms = sorted(excl_terms, key=len, reverse = True)                
                for term in excl_terms:
                    if bool(re.search(term, text)):
                        excl_flag = True
                        break
                    else:
                        excl_flag = False   
                            
                # Evaluating condition #2
                if (incl_flag == True) & (excl_flag == False):
                    cond2_flag = True
                else:
                    cond2_flag = False     

    else:
        cond2_flag = False            
#-------------------------------------------------- Evaluating condition #2 ---
 
    # Predicing
    if (cond1_flag == True) & (cond2_flag == True):
        y_hat = 1
    else:
        y_hat = 0
    
    return y_hat

###################################################### Declaring Functions ####


### Main routine ##############################################################
# Registering initial time
a = time.time()    
print("--start--")

# Open input file
datasets = pd.read_pickle(input_file)

columns = ['n', 'DB', 'Level', 'Column',
           'n_0', 'n_1',
           'Sensitivity Train (95% CI)', 'Specificity Train (95% CI)', 'AUC Train (95% CI)',
           'Sensitivity Validation (95% CI)', 'Specificity Validation (95% CI)', 'AUC Validation (95% CI)',
           'Sensitivity Test', 'Specificity Test', 'AUC Test',
           'Best_Classifier', 'Best_Parameters'
           ]

output_summary = pd.DataFrame(columns = columns)

cat_incl = pd.read_excel(aux_data, 'cat_inclusion') 
cat_excl = pd.read_excel(aux_data, 'cat_exclusion') 
sub_cat_incl = pd.read_excel(aux_data, 'sub-cat_inclusion') 
sub_cat_excl = pd.read_excel(aux_data, 'sub-cat_exclusion')

for n in range(1, 157): 
    print()
    print('Processing dataset number: ',n)
 
    # Loading dataset info
    dataset_info = datasets['info'].loc[n,:]
    n_0 = dataset_info['n_0']
    n_1 = dataset_info['n_1']
    db_info = dataset_info['data_option']
    level_info = dataset_info['level']
    column_info = dataset_info['column']
    go_on = dataset_info['go_on']
    
    # Continuing analysis if go_on == True
    if go_on == True:   
        dataset = datasets[n]    
        X_train = dataset['X_train'].reset_index()
        X_train_params = dataset['X_train_params'].reset_index()
        X_train = X_train.merge(right = X_train_params, on = ['index'])
        X_train.drop(['index'], axis = 1, inplace = True)
        del X_train_params
        
        X_validation = dataset['X_validation'].reset_index() 
        X_validation_params = dataset['X_validation_params'].reset_index()
        X_validation = X_validation.merge(right = X_validation_params, on = ['index'])
        X_validation.drop(['index'], axis = 1, inplace = True)
        del X_validation_params
                
        X_test = dataset['X_test'].reset_index()    
        X_test_params = dataset['X_test_params'].reset_index()       
        X_test = X_test.merge(right = X_test_params, on = ['index'])
        X_test.drop(['index'], axis = 1, inplace = True)
        del X_test_params

        y_train = dataset['y_train'].reset_index(drop = True)      
        y_validation = dataset['y_validation'].reset_index(drop = True)    
        y_test = dataset['y_test'].reset_index(drop = True)    
        
        # Predicting y_hat_train
        y_hat_train = X_train.apply(lambda x: predict(x, column_info), axis = 1)
        
        # Predicting y_hat_validation
        y_hat_validation = X_validation.apply(lambda x: predict(x, column_info), axis = 1)
        
        # Predicting y_hat_test
        y_hat_test = X_test.apply(lambda x: predict(x, column_info), axis = 1)
                
        # evaluating performance of train, validation and test sets
        sensitivity_train = '{:1.3f}'.format(round(recall_score(y_train, y_hat_train), 3))
        specificity_train = '{:1.3f}'.format(round(specifitiy(y_train, y_hat_train), 3))
        AUC_train = '{:1.3f}'.format(round(roc_auc_score(y_train, y_hat_train), 3))
        
        sensitivity_validation = '{:1.3f}'.format(round(recall_score(y_validation, y_hat_validation), 3))
        specificity_validation = '{:1.3f}'.format(round(specifitiy(y_validation, y_hat_validation), 3))
        AUC_validation = '{:1.3f}'.format(round(roc_auc_score(y_validation, y_hat_validation), 3))

        sensitivity_test = '{:1.3f}'.format(round(recall_score(y_test, y_hat_test), 3))
        specificity_test = '{:1.3f}'.format(round(specifitiy(y_test, y_hat_test), 3))
        AUC_test = '{:1.3f}'.format(round(roc_auc_score(y_test, y_hat_test), 3))              
           
        # Registering results
        output_summary.loc[n,'n'] = n
        output_summary.loc[n,'DB'] = db_info
        output_summary.loc[n,'Level'] = level_info
        output_summary.loc[n,'Column'] = column_info
        output_summary.loc[n,'n_0'] = n_0
        output_summary.loc[n,'n_1'] = n_1
        output_summary.loc[n,'Sensitivity Train (95% CI)'] = sensitivity_train
        output_summary.loc[n,'Specificity Train (95% CI)'] = specificity_train
        output_summary.loc[n,'AUC Train (95% CI)'] = AUC_train
        output_summary.loc[n,'Sensitivity Validation (95% CI)'] = sensitivity_validation
        output_summary.loc[n,'Specificity Validation (95% CI)'] = specificity_validation
        output_summary.loc[n,'AUC Validation (95% CI)'] = AUC_validation
        output_summary.loc[n,'Sensitivity Test'] = sensitivity_test
        output_summary.loc[n,'Specificity Test'] = specificity_test
        output_summary.loc[n,'AUC Test'] = AUC_test
        output_summary.loc[n,'Best_Classifier'] = 'Rule Based Classifier'
        output_summary.loc[n,'Best_Parameters'] = 'N/A'
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
        
    output_summary.to_excel(output_file)
   
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
