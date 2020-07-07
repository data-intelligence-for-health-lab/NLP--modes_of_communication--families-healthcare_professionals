###############################################################################
#                                                                             #
#                        machine learning approach part 1                     #
#  logistic regression,support vector machine,random forest,adaptive boosting #
#                                                                June 23 2020 #
###############################################################################



### Loading libraries #########################################################
import time
import numpy as np
seed = np.random.seed(42)
import pandas as pd
pd.options.mode.chained_assignment = None
import nltk
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, recall_score, confusion_matrix, roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from scipy import stats
import math 
import pickle
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
######################################################## Loading libraries ####



### Declaring I/O variables ###################################################
input_file = 'pre-processed_data.pickle'
output_file = 'ML_summary_part1.pickle'
################################################## Declaring I/O variables ####



### Declaring Functions #######################################################
def specifitiy(y, y_pred):
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    return (tn / (tn + fp))

class GridSearch4Pipes:
    def __init__(self, models, params):
        if not set(models.keys()).issubset(set(params.keys())):
            missing_params = list(set(models.keys()) - set(params.keys()))
            raise ValueError("Missing parameters: %s" % missing_params)
        self.pipes = models
        self.pipe_params = params
        self.keys = models.keys()
        self.grid_searches = {}

    def fit(self, X, y, cv, n_jobs = -1, verbose = 2, refit = 'AUC'): 
        scoring = {'sensitivity' : make_scorer(recall_score), 
                   'specificity' : make_scorer(specifitiy),
                   'AUC' : make_scorer(roc_auc_score)
                   }
        
        for key in self.keys:
            print()
            print("###########################################################")
            print('n: ', n)
            print("Data: ", db_info)
            print("Level: ", level_info)
            print("Column: ", column_info)
            print("Running GridSearchCV for %s." % key)
            print("***********************************************************")
            print()
            print()

            model = self.pipes[key]
            params = self.pipe_params[key]
            gs = GridSearchCV(model, params, cv = cv, n_jobs = n_jobs,
                              verbose = verbose, scoring = scoring, 
                              refit = refit, return_train_score = True, iid = False)
            gs.fit(X,y)
            self.grid_searches[key] = gs 
            
    def predict(self, X): 
        gs = self.grid_searches[self.master_key]
        y_pred = gs.predict(X)
        return y_pred
            
    def score_summary(self, sort_by='AUC Validation (95% CI)'):
        def row(key, s_specificity_train, s_AUC_train, s_sensitivity_validation, s_specificity_validation, s_AUC_validation, params):

            # sensitivity train              
            sensitivity_mean_train = '{:1.3f}'.format(round(np.mean(s_sensitivity_train), 3))
            sensitivity_LB_train = np.mean(s_sensitivity_train) - stats.t.ppf(1-0.025, kfold - 1)*np.std(s_sensitivity_train)/math.sqrt(kfold)
            if sensitivity_LB_train < 0:
                sensitivity_LB_train = 0            
            sensitivity_UB_train = np.mean(s_sensitivity_train) + stats.t.ppf(1-0.025, kfold - 1)*np.std(s_sensitivity_train)/math.sqrt(kfold)
            if sensitivity_UB_train > 1:
                sensitivity_UB_train = 1
            sensitivity_LB_train = '{:1.3f}'.format(sensitivity_LB_train,3)
            sensitivity_UB_train = '{:1.3f}'.format(sensitivity_UB_train,3)
            
            # sensitivity validation 
            sensitivity_mean_validation = '{:1.3f}'.format(round(np.mean(s_sensitivity_validation), 3))
            sensitivity_LB_validation = np.mean(s_sensitivity_validation) - stats.t.ppf(1-0.025, kfold - 1)*np.std(s_sensitivity_validation)/math.sqrt(kfold)
            if sensitivity_LB_validation < 0:
                sensitivity_LB_validation = 0             
            sensitivity_UB_validation = np.mean(s_sensitivity_validation) + stats.t.ppf(1-0.025, kfold - 1)*np.std(s_sensitivity_validation)/math.sqrt(kfold)
            if sensitivity_UB_validation > 1:
                sensitivity_UB_validation = 1
            sensitivity_LB_validation = '{:1.3f}'.format(sensitivity_LB_validation,3)                
            sensitivity_UB_validation = '{:1.3f}'.format(sensitivity_UB_validation,3)            

            # Specificity train
            specificity_mean_train = '{:1.3f}'.format(round(np.mean(s_specificity_train), 3))
            specificity_LB_train = np.mean(s_specificity_train) - stats.t.ppf(1-0.025, kfold - 1)*np.std(s_specificity_train)/math.sqrt(kfold)
            if specificity_LB_train < 0:
                specificity_LB_train = 0   
            specificity_UB_train = np.mean(s_specificity_train) + stats.t.ppf(1-0.025, kfold - 1)*np.std(s_specificity_train)/math.sqrt(kfold)
            if specificity_UB_train > 1:
                specificity_UB_train = 1
            specificity_LB_train = '{:1.3f}'.format(specificity_LB_train,3)
            specificity_UB_train = '{:1.3f}'.format(specificity_UB_train,3)
            
            # Specificity validation
            specificity_mean_validation = '{:1.3f}'.format(round(np.mean(s_specificity_validation), 3))
            specificity_LB_validation = np.mean(s_specificity_validation) - stats.t.ppf(1-0.025, kfold - 1)*np.std(s_specificity_validation)/math.sqrt(kfold)
            if specificity_LB_validation < 0:
                specificity_LB_validation = 0 
            specificity_UB_validation = np.mean(s_specificity_validation) + stats.t.ppf(1-0.025, kfold - 1)*np.std(s_specificity_validation)/math.sqrt(kfold)
            if specificity_UB_validation > 1:
                specificity_UB_validation = 1
            specificity_LB_validation = '{:1.3f}'.format(specificity_LB_validation,3) 
            specificity_UB_validation = '{:1.3f}'.format(specificity_UB_validation,3)            
            
            # AUC train
            AUC_mean_train = '{:1.3f}'.format(round(np.mean(s_AUC_train), 3))
            AUC_LB_train = np.mean(s_AUC_train) - stats.t.ppf(1-0.025, kfold - 1)*np.std(s_AUC_train)/math.sqrt(kfold)
            if AUC_LB_train < 0:
                AUC_LB_train = 0 
            AUC_UB_train = np.mean(s_AUC_train) + stats.t.ppf(1-0.025, kfold - 1)*np.std(s_AUC_train)/math.sqrt(kfold)
            if AUC_UB_train > 1:
                AUC_UB_train = 1
            AUC_LB_train = '{:1.3f}'.format(AUC_LB_train,3)
            AUC_UB_train = '{:1.3f}'.format(AUC_UB_train,3)
            
            # AUC validation
            AUC_mean_validation = '{:1.3f}'.format(round(np.mean(s_AUC_validation), 3))
            AUC_LB_validation = np.mean(s_AUC_validation) - stats.t.ppf(1-0.025, kfold - 1)*np.std(s_AUC_validation)/math.sqrt(kfold)
            if AUC_LB_validation < 0:
                AUC_LB_validation = 0
            AUC_UB_validation = np.mean(s_AUC_validation) + stats.t.ppf(1-0.025, kfold - 1)*np.std(s_AUC_validation)/math.sqrt(kfold)
            if AUC_UB_validation > 1:
                AUC_UB_validation = 1
            AUC_LB_validation = '{:1.3f}'.format(AUC_LB_validation,3)                   
            AUC_UB_validation = '{:1.3f}'.format(AUC_UB_validation,3)            
            
            d = {
                 'n' : n,
                 'Data' : db_info,
                 'Level' : level_info,
                 'Column' : column_info,
                 'Classifier': key,
                 'Parameters' : params,
                 'Sensitivity Train (95% CI)': sensitivity_mean_train+' ('+sensitivity_LB_train+'-'+sensitivity_UB_train+')',
                 'Specificity Train (95% CI)': specificity_mean_train+' ('+specificity_LB_train+'-'+specificity_UB_train+')',
                 'AUC Train (95% CI)': AUC_mean_train+' ('+AUC_LB_train+'-'+AUC_UB_train+')',   
                 'Sensitivity Validation (95% CI)': sensitivity_mean_validation+' ('+sensitivity_LB_validation+'-'+sensitivity_UB_validation+')',
                 'Specificity Validation (95% CI)': specificity_mean_validation+' ('+specificity_LB_validation+'-'+specificity_UB_validation+')',
                 'AUC Validation (95% CI)': AUC_mean_validation+' ('+AUC_LB_validation+'-'+AUC_UB_validation+')',                  
            }
            return pd.Series({**d})     

        rows = []        
        for k in self.grid_searches:
            params = self.grid_searches[k].cv_results_['params']
            sensitivity_scores_train = []
            specificity_scores_train = []
            AUC_scores_train = []
            sensitivity_scores_validation = []
            specificity_scores_validation = []
            AUC_scores_validation = []            

            for i in range(self.grid_searches[k].cv):
                key_sensitivity_train = "split{}_train_sensitivity".format(i)
                key_specificity_train = "split{}_train_specificity".format(i)
                key_AUC_train = "split{}_train_AUC".format(i) 
                key_sensitivity_validation = "split{}_test_sensitivity".format(i)
                key_specificity_validation = "split{}_test_specificity".format(i)
                key_AUC_validation = "split{}_test_AUC".format(i) 

                r_sensitivity_train = self.grid_searches[k].cv_results_[key_sensitivity_train ]
                r_specificity_train  = self.grid_searches[k].cv_results_[key_specificity_train ]
                r_AUC_train  = self.grid_searches[k].cv_results_[key_AUC_train ]     
                r_sensitivity_validation  = self.grid_searches[k].cv_results_[key_sensitivity_validation]
                r_specificity_validation = self.grid_searches[k].cv_results_[key_specificity_validation]
                r_AUC_validation = self.grid_searches[k].cv_results_[key_AUC_validation] 
                
                sensitivity_scores_train.append(r_sensitivity_train.reshape(len(params),1))
                specificity_scores_train.append(r_specificity_train.reshape(len(params),1))
                AUC_scores_train.append(r_AUC_train.reshape(len(params),1))  
                sensitivity_scores_validation.append(r_sensitivity_validation.reshape(len(params),1))
                specificity_scores_validation.append(r_specificity_validation.reshape(len(params),1))
                AUC_scores_validation.append(r_AUC_validation.reshape(len(params),1)) 

            
            sensitivity_scores_train = np.hstack(sensitivity_scores_train)
            specificity_scores_train = np.hstack(specificity_scores_train)
            AUC_scores_train = np.hstack(AUC_scores_train)              
            sensitivity_scores_validation = np.hstack(sensitivity_scores_validation)
            specificity_scores_validation = np.hstack(specificity_scores_validation)
            AUC_scores_validation = np.hstack(AUC_scores_validation)             
            
            for p, s_sensitivity_train, s_specificity_train, s_AUC_train, s_sensitivity_validation, s_specificity_validation, s_AUC_validation in zip(params, sensitivity_scores_train, specificity_scores_train, AUC_scores_train, sensitivity_scores_validation, specificity_scores_validation, AUC_scores_validation):
                rows.append((row(k, s_specificity_train, s_AUC_train, s_sensitivity_validation, s_specificity_validation, s_AUC_validation, p)))

        df = pd.concat(rows, axis=1, sort = False).T.sort_values([sort_by], ascending=False)
        df = df.reset_index(drop = True)
        
        # Setting order of preference if there is a tie
        # Order is based on faster processing times (data not shown)
        top_auc = df.loc[0,'AUC Validation (95% CI)']        
        df_top = df[df['AUC Validation (95% CI)'] == top_auc]
        list_classifiers = list(df_top['Classifier'].unique())     

        if 'LogisticRegression' in list_classifiers:
            top_classifier = 'LogisticRegression'
        elif 'SVC' in list_classifiers:
            top_classifier = 'SVC'
        elif 'AdaBoostClassifier' in list_classifiers:
            top_classifier = 'AdaBoostClassifier'
        else:
            top_classifier = 'RandomForestClassifier'
        df_top = df_top[df_top['Classifier'] == top_classifier].reset_index(drop = True)                
        df_top = df_top.loc[0,:]
        
        self.master_key = df_top['Classifier']
       
        return df_top


###################################################### Declaring Functions ####



### Main routine ##############################################################
# Registering initial time
a = time.time()    
print("--start--")

# Open input file
datasets = pd.read_pickle(input_file)

kfold = 10

columns = ['n', 'DB', 'Level', 'Column',
           'n_0', 'n_1',
           'Sensitivity Train (95% CI)', 'Specificity Train (95% CI)', 'AUC Train (95% CI)',
           'Sensitivity Validation (95% CI)', 'Specificity Validation (95% CI)', 'AUC Validation (95% CI)',
           'Sensitivity Test', 'Specificity Test', 'AUC Test',
           'Best_Classifier', 'Best_Parameters'
           ]

output_summary = pd.DataFrame(columns = columns)

n_datasets = len(datasets['info'])

# Classifiers
lr = LogisticRegression(penalty = 'none', class_weight = 'balanced', max_iter = 1e4, random_state = seed, solver = 'saga')
svc = SVC(class_weight = 'balanced', max_iter = 1e4, random_state = seed, kernel = 'linear')
rfc = RandomForestClassifier(random_state = seed, class_weight = 'balanced')
abc = AdaBoostClassifier(random_state = seed)


# Classifiers pipelines
pipeline_lr = Pipeline(steps=[('tfidf', TfidfVectorizer()),
                              ('LogisticRegression', lr)])
        
pipeline_svc = Pipeline(steps=[('tfidf', TfidfVectorizer()),
                               ('SVC', svc)])   
        
pipeline_rfc = Pipeline(steps=[('tfidf', TfidfVectorizer()),
                               ('RandomForestClassifier', rfc)])   
        
pipeline_abc = Pipeline(steps=[('tfidf', TfidfVectorizer()),
                               ('AdaBoostClassifier', abc)])         
        

pipe_models = {'LogisticRegression' : pipeline_lr,
                'SVC': pipeline_svc,
                'RandomForestClassifier': pipeline_rfc,
                'AdaBoostClassifier': pipeline_abc
               }


pipe_params = {
        'LogisticRegression' : {
            'tfidf__ngram_range' : [(1,1), (1,2), (1,3)],
                'tfidf__max_df' : [0.7, 0.8, 0.9, 0.95, 1.0],
                'tfidf__min_df' : [2, 10, 50],
                'tfidf__binary' : [False, True],
                'tfidf__use_idf' : [False, True],
                'tfidf__norm' : ['l1', 'l2', None],
                      },
         
            'SVC': {
                'tfidf__ngram_range' : [(1,1), (1,2), (1,3)],
                   'tfidf__max_df' : [0.7, 0.8, 0.9, 0.95, 1.0],
                   'tfidf__min_df' : [2, 10, 50],
                   'tfidf__binary' : [False, True],
                   'tfidf__use_idf' : [False, True],
                   'tfidf__norm' : ['l1', 'l2', None],                 
                    },
         
            'RandomForestClassifier' : {
                'tfidf__ngram_range' : [(1,1), (1,2), (1,3)],
                   'tfidf__max_df' : [0.7, 0.8, 0.9, 0.95, 1.0],
                   'tfidf__min_df' : [2, 10, 50],
                   'tfidf__binary' : [False, True],
                   'tfidf__use_idf' : [False, True],
                   'tfidf__norm' : ['l1', 'l2', None],
                    },
         
              'AdaBoostClassifier' : {
                  'tfidf__ngram_range' : [(1,1), (1,2), (1,3)],
                    'tfidf__max_df' : [0.7, 0.8, 0.9, 0.95, 1.0],
                    'tfidf__min_df' : [2, 10, 50],
                    'tfidf__binary' : [False, True],
                    'tfidf__use_idf' : [False, True],
                    'tfidf__norm' : ['l1', 'l2', None],
                      },         
} 

gs = GridSearch4Pipes(pipe_models, pipe_params)

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
  
    if go_on == True:   
        dataset = datasets[n]    
        X_train_validation = dataset['X_train_validation']
        y_train_validation = dataset['y_train_validation']
        
        X_test = dataset['X_test']
        y_test = dataset['y_test']
            
        # Fit best composition of parameters acording to gridsearch
        gs.fit(X_train_validation, y_train_validation, cv = kfold)
                
        # Calculate train/validation metrics
        fit_summary = gs.score_summary()
        
        # Predict y_test
        y_pred_test = gs.predict(X_test)

        # Evaluating validation prediction
        sensitivity_test = recall_score(y_test, y_pred_test)
        specificity_test = specifitiy(y_test, y_pred_test)
        AUC_test = roc_auc_score(y_test, y_pred_test)
         
        output_summary.loc[n,'n'] = n
        output_summary.loc[n,'DB'] = db_info
        output_summary.loc[n,'Level'] = level_info
        output_summary.loc[n,'Column'] = column_info
        output_summary.loc[n,'n_0'] = n_0
        output_summary.loc[n,'n_1'] = n_1
        output_summary.loc[n,'Sensitivity Train (95% CI)'] = fit_summary['Sensitivity Train (95% CI)']
        output_summary.loc[n,'Specificity Train (95% CI)'] = fit_summary['Specificity Train (95% CI)']
        output_summary.loc[n,'AUC Train (95% CI)'] = fit_summary['AUC Train (95% CI)']
        output_summary.loc[n,'Sensitivity Validation (95% CI)'] = fit_summary['Sensitivity Validation (95% CI)']
        output_summary.loc[n,'Specificity Validation (95% CI)'] = fit_summary['Specificity Validation (95% CI)']
        output_summary.loc[n,'AUC Validation (95% CI)'] = fit_summary['AUC Validation (95% CI)']   
        output_summary.loc[n,'Sensitivity Test'] = '{:1.3f}'.format(round(sensitivity_test,3))
        output_summary.loc[n,'Specificity Test'] = '{:1.3f}'.format(round(specificity_test,3))
        output_summary.loc[n,'AUC Test'] = '{:1.3f}'.format(round(AUC_test,3))
        output_summary.loc[n,'Best_Classifier'] = fit_summary['Classifier']
        output_summary.loc[n,'Best_Parameters'] = str(fit_summary['Parameters'])

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
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    print('          Saving results for dataset number: ', n,)
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    print()

# Registering final time
b = time.time()        
print('--end--')
print('Total processing time: %0.2f minutos' %((b-a)/60))    
############################################################# Main routine ####
