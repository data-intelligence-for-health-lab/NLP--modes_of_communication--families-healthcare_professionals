###############################################################################
#                                                                             #
#                 pre-processing and dataset construction                     #
#                                                                 July 6 2020 #
###############################################################################



### Loading libraries #########################################################
import os
import pickle
import pandas as pd
import numpy as np
import re
import string
import time
import nltk
from nltk.stem import WordNetLemmatizer 
from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
######################################################## Loading libraries ####



### Declaring I/O variables ###################################################
input_file = 'input_file.xlsx'
output_file = 'pre-processed_data.pickle'
output_summary = 'pre-processed_data_summary.xlsx'
aux_data = 'auxiliar_data.xlsx'
names_pickle = "names_dict.pickle"
################################################## Declaring I/O variables ####



### Declaring Functions #######################################################
def load_NER():
    # NER tagger should be installed prior to use.
    # Please refer to https://nlp.stanford.edu/software/CRF-NER.html
    java_path = "C:\\Program Files\\Java\\jdk-12.0.2\\bin\\java.exe"
    os.environ['JAVAHOME'] = java_path
    st = StanfordNERTagger('G:\\API\\Stanford\\stanford-ner-2018-10-16\\classifiers\\english.all.3class.distsim.crf.ser.gz',
					       'G:\\API\\Stanford\\stanford-ner-2018-10-16\\stanford-ner.jar',
					        encoding='utf-8')
    return st

def mount_dict(file, sheet):
    df = pd.read_excel(file, sheet_name = sheet) 
    key = list(df)[0]
    meaning = list(df)[1]
    df.index = df[key].str.len()
    df = df.sort_index(ascending = False).reset_index(drop=True)
    df = df.set_index(key).to_dict()[meaning]
    return df

def apply_dict(text, dictionary):
    for word in dictionary.keys():
        text = re.sub("(?<![a-zA-Z0-9])"+word+"(?![a-zA-Z0-9])",
                      dictionary[word],
                      text,
                      flags=re.I)    
    return text

def lemmatization(text):
    WNL = WordNetLemmatizer()
    tokenized_text = nltk.word_tokenize(text)    
    return ' '.join([WNL.lemmatize(word) for word in tokenized_text])

def mount_list(file, sheet):
    df = pd.read_excel(file, sheet_name = sheet) 
    ref_label = list(df)[0]
    df.index = df[ref_label].str.len()
    df = df.sort_index(ascending = False).reset_index(drop=True)
    return df[ref_label].tolist()

def mount_lowercasing_list():
    WNL = WordNetLemmatizer()
    relations_list = list(mount_dict(aux_data,'relations&genders').keys())
    countries_list = mount_list(aux_data,'lower_country')
    nationalities_list = mount_list(aux_data,'lower_nationality')
    languages_list = mount_list(aux_data,'lower_language')
    cities_list = mount_list(aux_data,'lower_city')
    provinces_list = mount_list(aux_data,'lower_province')
    miscellania_list = mount_list(aux_data,'lower_miscellania') 
    lower_stopwords_list = mount_list(aux_data,'lower_stopwords') 
    lower_calendar_list = mount_list(aux_data,'lower_calendar')    
    lowercasing_list = relations_list + countries_list + nationalities_list + languages_list + cities_list + provinces_list + miscellania_list + lower_stopwords_list + lower_calendar_list
    lowercasing_list = sorted(lowercasing_list, key = len, reverse  = True)
    lowercasing_list = [WNL.lemmatize(word.lower()) for word in lowercasing_list]
    return lowercasing_list

def lowercase(text, lowercasing_list):
    for word in lowercasing_list:
        text = re.sub('(?<![a-zA-Z0-9])'+word+'(?![a-zA-Z0-9])',
                      word.lower(),
                      text,
                      flags=re.I) 
    return text 

def search4names(text):
    # Open pickle with names
    with open(names_pickle, 'rb') as handle:
        names_dict = pickle.load(handle)
    names = list(names_dict.keys())
    new_text = []
    NER = load_NER()
    tokenized_text = nltk.word_tokenize(text)
    tagged_text = NER.tag(tokenized_text)
    
    # title case names and lowercase other terms
    for word, tag in tagged_text:
        if (tag == "PERSON") or (word.title() in names):
            new_text.append(word.title())
        else:
            new_text.append(word.lower())
    text = ' '.join(new_text)    
    
    # Register which tokens are names (trigrams)   
    tokenized_text = nltk.word_tokenize(text)   
    for w1,w2,w3 in nltk.trigrams(tokenized_text):
        if ((len(w1) > 1) & (len(w2) > 1) & (len(w3) > 1)):
            if ((w1[0].isupper()) & (w1[1].islower())) & \
                ((w2[0].isupper()) & (w2[1].islower())) & \
                ((w3[0].isupper()) & (w3[1].islower())) & \
                (w1 in names):
                    gender = names_dict[w1]
                    text = re.sub('(?<![a-zA-Z0-9])'+w1+' '+w2+' '+w3+'(?![a-zA-Z0-9])', \
                                  '|N|'+w1+'_'+w2+'_'+w3+'|'+str(gender)+'|', \
                                  text)
            elif ((w1[0].isupper()) & (w1[1].islower())) & \
                  ((w2[0].isupper()) & (w2[1].islower())) & \
                  ((w3[0].isupper()) & (w3[1].islower())):
                  text = re.sub('(?<![a-zA-Z0-9])'+w1+' '+w2+' '+w3+'(?![a-zA-Z0-9])', \
                                '|N|'+w1+'_'+w2+'_'+w3+'|U|', \
                                text)
    
    # Register which tokens are names (bigrams)                    
    tokenized_text = nltk.word_tokenize(text)
    for w1,w2 in nltk.bigrams(tokenized_text):
        if ((len(w1) > 1) & (len(w2) > 1)):
            if ((w1[0].isupper()) & (w1[1].islower())) & \
                ((w2[0].isupper()) & (w2[1].islower())) & \
                (w1 in names):
                    gender = names_dict[w1]                   
                    text = re.sub('(?<![a-zA-Z0-9])'+w1+' '+w2+'(?![a-zA-Z0-9])', \
                                  '|N|'+w1+'_'+w2+'|'+str(gender)+'|', \
                                  text)
            elif ((w1[0].isupper()) & (w1[1].islower())) & \
                  ((w2[0].isupper()) & (w2[1].islower())):
                  text = re.sub('(?<![a-zA-Z0-9])'+w1+' '+w2+'(?![a-zA-Z0-9])', \
                                  '|N|'+w1+'_'+w2+'|U|', \
                                   text)
                      
    # Register which tokens are names (unigrams)                       
    tokenized_text = nltk.word_tokenize(text)
    for w1 in tokenized_text:
        if (len(w1) > 1):
            if ((w1[0].isupper()) & (w1[1].islower())):
                if (w1 in names):  
                    gender = names_dict[w1]                
                    text = re.sub('(?<!(\w)|\|)'+w1+'(?!(\w)|\|)','|N|'+w1+'|'+str(gender)+'|',text)
                else:
                    text = re.sub('(?<!(\w)|\|)'+w1+'(?!(\w)|\|)', '|N|'+w1+'|U|', text)
    return text

def search4relations(text):
    relations_list = mount_list(aux_data,'relations&genders')
    WNL = WordNetLemmatizer()
    relations_list = [WNL.lemmatize(word.lower()) for word in relations_list]
    for word in relations_list:
        text = re.sub('(?<!(\w)|\|)'+word+'(?!(\w)|\|)',
                      '|R|'+word+'|r|',
                      text,
                      flags=re.I) 
    return text

def remove_untagged(x):
    text = [word if '|N|' in word else \
            word if '|R|' in word else \
            word if '.' in word else "" for word in nltk.word_tokenize(x)]
    text = ' '.join([word for word in text if len(word) > 0])
    return text

def extract_info(text, record_id):
    # Identify the sequence that info occurs
    TokenList = nltk.word_tokenize(text)
    Sequence = [] #list of tuples ((0:name,1:relation,2:'.'), tokenlist position)
    for t in range(0,len(TokenList)):
        token = TokenList[t]
        if '|N|' in token:
            Sequence.append((0,t))
        elif '|R|' in token:
            Sequence.append((1,t))
        elif ('.' in token) or (',' in token):
            Sequence.append((2,t))  
    NoInfo = len(Sequence)        
    
    # Identifies in which token occurs a "break". A break is when the next info
    # refers to other combination of name/relation.
    BreakIndex = [0]
    FlagN = 0
    FlagR = 0
    for n in range(0,NoInfo):
        Categorie = Sequence[n][0]
        if (Categorie == 2) & (n != (NoInfo-1)):
            FlagN = 0
            FlagR = 0
            BreakIndex.append(n)
        elif (Categorie == 0) & (FlagN == 1):
            FlagR = 0
            BreakIndex.append(n)
        if (Categorie == 0) & (FlagN == 0):
            FlagN = 1
        elif (Categorie == 0) & (FlagN == 1):
            FlagR = 0
            BreakIndex.append(n)
        elif (Categorie == 1) & (FlagR == 0):
            FlagR = 1              
        elif (Categorie == 1) & (FlagR == 1):
            FlagN = 0
            BreakIndex.append(n)
    if BreakIndex[-1] != NoInfo:
        BreakIndex.append(NoInfo)
    Breaks = [[x,y] for x,y in zip(BreakIndex, BreakIndex[1:])]
    
    TempDF = pd.DataFrame(columns=['Record_ID','Name','Relation'])
    
    # Create one line for each name/relation identified. 
    for n in range(0,len(Breaks)):
        Start = Breaks[n][0]
        End = Breaks[n][1]
        Slice = Sequence[Start:End]
        Categories = []
        for [Categorie,Index] in Slice:
            Categories.append(Categorie)
        
        # Avoiding slices with only punct
        if ((2 in Categories) & (0 not in Categories) & (1 not in Categories)): 
            if n != 0:
                Breaks[n-1][1] = Breaks[n][1]
                Breaks[n] = Breaks[n-1] 
                
    # Removing duplicate slices (they occur if there were slices with only punct
    NewBreaks = []
    for i in Breaks:
        if i not in NewBreaks:
            NewBreaks.append(i)
    Breaks = NewBreaks
    for n in range(0,len(Breaks)):
        Start = Breaks[n][0]
        End = Breaks[n][1]
        Slice = Sequence[Start:End]
        Slice.sort(key=lambda x: x[0])
        for [Categorie,Index] in Slice:
            if Categorie == 0:
                TempDF.loc[n,'Name'] = TokenList[Index]
            if Categorie == 1:
                TempDF.loc[n,'Relation'] = TokenList[Index]
            TempDF.loc[n,'Record_ID'] = record_id
    return TempDF 

def adjust_info(table):
    # Identifying male and female relations
    relations_dict = mount_dict(aux_data,'relations&genders')
    relations_dict = pd.DataFrame.from_dict(relations_dict, 
                                            orient = 'index', 
                                            columns = ['gender'])
    relations_dict.reset_index(inplace = True)
    male_relations_list = list(relations_dict[relations_dict['gender'] == 'M']['index'])
    female_relations_list = list(relations_dict[relations_dict['gender'] == 'F']['index'])
    
    # Drop duplicate records
    table = table.drop_duplicates().reset_index(drop = True)

    # Delete prefixes and sufixes for analysis
    table['Name'] = table['Name'].map(lambda x: re.sub('(?<![a-zA-Z0-9])(nan)(?![a-zA-Z0-9])','|N|NA|U|',str(x)))    
    table['Name'] = table['Name'].map(lambda x: re.sub('(?<![0-9])_(?![0-9])',' ',str(x)))        
    table['Relation'] = table['Relation'].map(lambda x: re.sub('(?<![a-zA-Z0-9])(nan)(?![a-zA-Z0-9])','|R|NA|U|',str(x)))                         

    # Adjusting names
    for n in range(0,len(table)):
        Actual = table.loc[n,'Name'][3:-3]
        for m in range(0,len(table)):
            Other = table.loc[m,'Name'][3:-3]
            if (str(Actual) != '|N|NA|U|') & (str(Other) != '|N|NA|U|'):
                if Actual.find(Other) >= 0:
                    LenActual = len(Actual)
                    LenOther = len(Other)
                    if LenActual > LenOther:
                        table.loc[m,'Name'] = table.loc[n,'Name']    
    table = table.drop_duplicates().reset_index(drop = True)
  
    # Adjusting relation = NA when name already exist
    temp_table = pd.DataFrame()
    names_list = list(table['Name'].unique())
    for name in names_list:
        temp1 = table[table['Name'] == name]
        if len(temp1) > 1:
            temp2 = temp1[temp1['Relation'] != "|R|NA|U|"]
            temp_table = pd.concat([temp_table,temp2], axis = 0)
        else:
            temp_table = pd.concat([temp_table,temp1], axis = 0)            
    table = temp_table.copy()
    
    # Adjusting name = NA when relation already exist
    new_table = pd.DataFrame()
    relations_list = list(table['Relation'].unique())
    for relation in relations_list:
        temp1 = table[table['Relation'] == relation]
        if len(temp1) > 1:
            temp2 = temp1[temp1['Name'] != "|N|NA|U|"]
            new_table = pd.concat([new_table,temp2], axis = 0)
        else:
            new_table = pd.concat([new_table,temp1], axis = 0)
    table = new_table.reset_index(drop = True).copy()
    
    # Adjusting remaining relation = NA
    for n in range(0, len(table)):
        if table.loc[n,'Relation'] == '|R|NA|U|':
            table.loc[n,'Relation'] = '|R|other|r|'
    
    # Including gender on relation
    for n in range(0, len(table)):
        relation_temp = table.loc[n,'Relation']
        relation_temp_free = re.sub('(\|R\|)|(\|r\|)','',str(relation_temp))        
        if relation_temp_free in male_relations_list:
            gender = 'M'
        elif relation_temp_free in female_relations_list:
            gender = 'F'
        else:
            gender_letter = table.loc[n,'Name'][-2:-1]
            if gender_letter == 'F':
                gender = 'F'
            elif gender_letter == 'M':
                gender = 'M'
            else:
                gender = 'U'
        relation = relation_temp[:-2]+gender+'|'
        table.loc[n,'Relation'] = relation        
    
    # Delete remaining prefixes and sufixes (names)
    table['Name'] = table['Name'].map(lambda x: re.sub('(\|N\|)|(\|n\|)|(\|M\|)|(\|F\|)|(\|U\|)','',str(x)))    
    return table    

def names2relations(x):
    # Identifying unknown gender in relations
    relations_dict = mount_dict(aux_data,'relations&genders')
    relations_dict = pd.DataFrame.from_dict(relations_dict, 
                                            orient = 'index', 
                                            columns = ['gender'])
    relations_dict.reset_index(inplace = True)
    unknown_relations_list = list(relations_dict[relations_dict['gender'] == 'U']['index'])    
    record_id = x['Record_ID']
    text = x['ChartValue']
   
    # Looking available info at net_table
    temp = net_table[(net_table['Record_ID'] == record_id) & 
                      (net_table['Name'] != 'NA')].reset_index(drop = True)    

    # Searching for each name individually. relation keeps prefixes and 
    # sufixes if the relation term doesn't tell gender
    if len(temp) > 0:
        for row in range(len(temp)):
            name = temp.loc[row, 'Name']
            relation01 = temp.loc[row, 'Relation']
            relation02 = relation01[3:-3]            
            if relation02 in unknown_relations_list:
                relation = relation01
            else:
                relation = relation02
            
            # Substituting name per relation                
            text = re.sub('(?<![a-zA-Z0-9])'+name+'(?![a-zA-Z0-9])',
                          relation,
                          text)    
    return text

def lower_exclude(x):    
    # lowercasing
    x = ' '.join([token.lower() for token in nltk.word_tokenize(x)])

    # excluding numbers
    x = re.sub('[0-9]', '', x)
    
    # excluding punctiation
    x = ' '.join([word for word in nltk.word_tokenize(x) if word not in set(string.punctuation)])
    return x

def split_data(temp):
    y_raw = temp['y'].copy()
    X_raw = temp['X'].copy().astype(str)    
    X_train_validation, X_test, y_train_validation, y_test = train_test_split(X_raw,
                                                                              y_raw, 
                                                                              test_size = 0.10, 
                                                                              random_state = 42, 
                                                                              stratify = y_raw)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train_validation,
                                                                    y_train_validation, 
                                                                    test_size = 0.10, 
                                                                    random_state = 42, 
                                                                    stratify = y_train_validation)
    return X_train_validation, X_train, X_validation, X_test, y_train_validation, y_train, y_validation, y_test

def get_params(X_train_validation, X_train, X_validation, X_test, temp):
    X_train_validation_index = list(X_train_validation.reset_index()['index'])
    X_train_index = list(X_train.reset_index()['index'])
    X_validation_index = list(X_validation.reset_index()['index'])
    X_test_index = list(X_test.reset_index()['index'])
    X_train_validation_params = temp[temp.index.isin(X_train_validation_index)]['ParameterID'].reindex(index = X_train_validation.index)
    X_train_params = temp[temp.index.isin(X_train_index)]['ParameterID'].reindex(index = X_train.index)
    X_validation_params = temp[temp.index.isin(X_validation_index)]['ParameterID'].reindex(index = X_validation.index)
    X_test_params = temp[temp.index.isin(X_test_index)]['ParameterID'].reindex(index = X_test.index)
    X_train_validation_ID = temp[temp.index.isin(X_train_validation_index)]['Record_ID'].reindex(index = X_train_validation.index)
    X_train_ID = temp[temp.index.isin(X_train_index)]['Record_ID'].reindex(index = X_train.index)
    X_validation_ID = temp[temp.index.isin(X_validation_index)]['Record_ID'].reindex(index = X_validation.index)
    X_test_ID = temp[temp.index.isin(X_test_index)]['Record_ID'].reindex(index = X_test.index)    
    return X_train_validation_params, X_train_params, X_validation_params, X_test_params, \
        X_train_validation_ID, X_train_ID, X_validation_ID, X_test_ID
###################################################### Declaring Functions ####



### Main routine ##############################################################
# Registering initial time
a = time.time()    
print("--start--")

# Open input file
data = pd.read_excel(io = input_file, sheet_name = 'Sheet1')

# Pre-processing steps, as presented in the original paper, are:
#--- Step 1 -------------------------------------------------------------------
print('Pre-Processing - Step #1...')

# Excluding duplicated data (same full record)
data = data.drop_duplicates().reset_index(drop = True)

# Ensuring  that field 'ChartValue' contains only strings
data['ChartValue'] = data['ChartValue'].apply(lambda x: str(x))

# adjust string => 'abc123' => 'abc 123'
data['ChartValue'] = data['ChartValue'].apply(
    lambda x: " ".join(
        re.sub(r"([0-9]+(\.[0-9]+)?)",
                r" \1 ",
                x).strip().split()))

# adjust string => 'abc-' => 'abc -'
data['ChartValue'] = data['ChartValue'].apply(
    lambda x: " ".join(
        x.translate(
            str.maketrans(
                {key: " {0} ".format(key) for key in string.punctuation})).split()))
#------------------------------------------------------------------ Step 1 ----

#--- Step 2 -------------------------------------------------------------------
print('Pre-Processing - Step #2...')
# Loading dictionary to standardize relations. E.g.: "sis" >> "sister"
relations_dict = mount_dict(aux_data,'standardize_relations')

# standardizes relations
data['ChartValue'] = data['ChartValue'].apply(
    lambda x: apply_dict(x, relations_dict))
#------------------------------------------------------------------ Step 2 ----

#--- Step 3 -------------------------------------------------------------------
print('Pre-Processing - Step #3...')
# spell checker step. On the original experiment, data was saved to a xlsx file
# and the spell checker from MS Excel was used to adjust text. 
# The following lines can be use to save and load data

# To save it, use:
# data.to_excel('data_spell_check.xlsx')

# To read it, use:
# data = pd.read_excel('data_spell_check.xlsx')
#------------------------------------------------------------------ Step 3 ----

#--- Step 4 -------------------------------------------------------------------
print('Pre-Processing - Step #4...')
# Loading dictionary to adjust indirect relations
indirect_relations_dict = mount_dict(aux_data,'indirect_relations_dict')

# standardizes relations
data['ChartValue'] = data['ChartValue'].apply(
    lambda x: apply_dict(x, indirect_relations_dict))
#------------------------------------------------------------------ Step 4 ----

#--- Step 5 -------------------------------------------------------------------
print('Pre-Processing - Step #5...')
# lemmatization of ChartValue
data['ChartValue'] = data['ChartValue'].apply(lambda x: lemmatization(x))
#------------------------------------------------------------------ Step 5 ----

#--- Step 6 -------------------------------------------------------------------
print('Pre-Processing - Step #6...')
# Filtering to keep only notes related to parameters:
#  4957: Family Quick View Summary
# 17202: Contact Information Family
net_data = data[(data['ParameterID'] == 4957) | 
                (data['ParameterID'] == 17202)].reset_index(drop=True).copy()

# Lowercase terms in lowercasing_list. It aims to help NER tager to correctly
# identify names in text
lowercasing_list = mount_lowercasing_list()
net_data['ChartValue'] = net_data['ChartValue'].apply(
    lambda x: lowercase(x, lowercasing_list))

# Searching for names
net_data['ChartValue'] = net_data['ChartValue'].apply(
    lambda x: search4names(x))

# Searching for relations 
net_data['ChartValue'] = net_data['ChartValue'].apply(
    lambda x: search4relations(x))

# Remove untagged tokens
net_data['ChartValue'] = net_data['ChartValue'].apply(
    lambda x: remove_untagged(x))

# Excluding records where no entities were recognized
net_data = net_data[net_data.ChartValue.apply(lambda x: len(str(x)) > 0)]
net_data = net_data.reset_index(drop = True)

# Dropping unused columns and duplicated rows
net_data = net_data[['Record_ID','ChartValue']]
net_data = net_data.drop_duplicates().reset_index(drop = True)

# Creating network table
net_table = pd.DataFrame(columns=['Record_ID','Name','Relation'])

# identify existing record_ids
record_ids = list(net_data['Record_ID'].unique())

# Analyzing one record_id at a time
for record_id in record_ids:
    temp = net_data[net_data['Record_ID'] == record_id].reset_index(drop = True)    
    new_temp = pd.DataFrame(columns=['Record_ID','Name','Relation'])

    # Analyzing each row of temp
    for row in range(len(temp)):
        text = temp.loc[row, 'ChartValue']
    
        # Extracting info available in each record
        extracted_info = extract_info(text, record_id)        
        new_temp = pd.concat([new_temp, extracted_info], ignore_index = True)
    
    # Normalize names. e.g.: line1: Joao, line2:Joao_Silva => all become Joao_Silva. Drops duplicate records
    new_temp = adjust_info(new_temp)   
    
    # Appending network table
    net_table = pd.concat([net_table, new_temp], ignore_index = True, sort = False) 
#------------------------------------------------------------------ Step 6 ----

#--- Step 7 -------------------------------------------------------------------
print('Pre-Processing - Step #7...')

# Substituing names per relation. Adding gender info in ambiguous relationships
data['ChartValue'] = data.apply(lambda x: names2relations(x), axis = 1)
#------------------------------------------------------------------ Step 7 ----

#--- Step 8 -------------------------------------------------------------------
print('Pre-Processing - Step #8...')

# Lowercasing records and excluding numbers and punctuation
data['ChartValue'] = data['ChartValue'].apply(lambda x: lower_exclude(x))
#------------------------------------------------------------------ Step 8 ----


#--- building datasets --------------------------------------------------------
print()
print('Building datasets..')
# adjusting data to macro granualarity
# text
chartvalue_macro = data.groupby('Record_ID')['ChartValue'].apply(' '.join)

# parameters
data['ParameterID'] = data['ParameterID'].apply(lambda x: str(x))
params_macro = data.groupby('Record_ID')['ParameterID'].apply(lambda x: ' '.join(set(x)))

# reference standard
reference_macro = data.groupby('Record_ID').sum()
reference_macro = reference_macro.apply(lambda x: x.apply(lambda y: 1 if y > 0 else 0), axis = 1)

# full macro data
macro_data = pd.concat([params_macro,chartvalue_macro,reference_macro], axis = 1).reset_index()

# initializing outputs 
output = {}
output_info = pd.DataFrame()
n = 1

# preparing one category at a time.
# F: Documented family or friends
# V: Visits
# P: Phone calls
categories = ['F', 'V', 'P']
for category in categories:
    print()
    print('Processing category: ', category)
    if category == 'F':
        columns = [column for column in data.columns if bool(re.search('F[0-9]', column))]
        for column in columns:
            print('Processing sub-category: ', column)
            output_temp = {}
            temp = macro_data.loc[:, ['Record_ID', 'ParameterID', column, 'ChartValue']].rename(columns = {column:'y', 'ChartValue':'X'})
            sum_n1 = temp['y'].sum()
            sum_n0 = len(temp) - sum_n1    
            # checking if there are at least 11 patients in each class 
            # just remembering that each line refers to a patient in macro granularity
            if sum_n1 >= 11:                
                X_train_validation, X_train, X_validation, X_test, y_train_validation, y_train, y_validation, y_test = split_data(temp)
                X_train_validation_params, X_train_params, X_validation_params, X_test_params, X_train_validation_ID, X_train_ID, X_validation_ID, X_test_ID = get_params(X_train_validation, X_train, X_validation, X_test, temp)
                
                train_validation_record_id = list(temp['Record_ID'][X_train_validation.index])
                train_record_id = list(temp['Record_ID'][X_train.index])
                validation_record_id = list(temp['Record_ID'][X_validation.index])
                test_record_id = list(temp['Record_ID'][X_test.index])
                go_on = True
            else:
                X_train = 'N/A'
                X_train_validation = 'N/A'
                X_validation = 'N/A'
                X_test = 'N/A'
                X_train_validation_params = 'N/A'
                X_train_params = 'N/A'
                X_validation_params = 'N/A'
                X_test_params = 'N/A'
                
                X_train_validation_ID = 'N/A'
                X_train_ID = 'N/A'
                X_validation_ID = 'N/A'
                X_test_ID = 'N/A'                    
                
                y_train = 'N/A'
                y_train_validation = 'N/A'
                y_validation = 'N/A'
                y_test = 'N/A'
                go_on = False

            output_temp['X_train'] = X_train
            output_temp['X_train_validation'] = X_train_validation
            output_temp['X_validation'] = X_validation
            output_temp['X_test'] = X_test
            
            output_temp['X_train_validation_params'] = X_train_validation_params
            output_temp['X_train_params'] = X_train_params
            output_temp['X_validation_params'] = X_validation_params
            output_temp['X_test_params'] = X_test_params          
            
            output_temp['X_train_validation_ID'] = X_train_validation_ID
            output_temp['X_train_ID'] = X_train_ID
            output_temp['X_validation_ID'] = X_validation_ID
            output_temp['X_test_ID'] = X_test_ID                  

            output_temp['y_train'] = y_train
            output_temp['y_train_validation'] = y_train_validation
            output_temp['y_validation'] = y_validation
            output_temp['y_test'] = y_test
            
            output[n] = output_temp
            
            info_temp = pd.DataFrame()
            info_temp.loc[0,'n'] = n
            info_temp.loc[0,'n_0'] = sum_n0
            info_temp.loc[0,'n_1'] = sum_n1                
            info_temp.loc[0,'data_option'] = 'F'            
            info_temp.loc[0,'level'] = 'macro'
            info_temp.loc[0,'column'] = column
            info_temp.loc[0,'go_on'] = go_on
            output_info = pd.concat([output_info, info_temp], axis = 0, ignore_index  = True)
            n += 1

    elif category == 'V':
        columns = [column for column in data.columns if bool(re.search('V[0-9]', column))]
        
        for column in columns:
            print('Processing sub-category: ', column)
            output_temp_l0 = {}
            output_temp_l1 = {}
            temp_l0 = macro_data.loc[:, ['Record_ID', 'ParameterID', column, 'ChartValue']].rename(columns = {column:'y', 'ChartValue':'X'})
            temp_l1 = data.loc[:, ['Record_ID', 'ParameterID', column, 'ChartValue']].rename(columns = {column:'y', 'ChartValue':'X'})
            sum_n1_l0 = temp_l0['y'].sum()
            sum_n0_l0 = len(temp_l0) - sum_n1_l0       
            
            sum_n1_l1 = temp_l1['y'].sum()
            sum_n0_l1 = len(temp_l1) - sum_n1_l1  
            
            # checking if there are at least 11 patients in each class 
            # just remembering that each line refers to a patient in sum_n1_l0
            if sum_n1_l0 >= 11:                
                X_train_validation_l0, X_train_l0, X_validation_l0, X_test_l0, y_train_validation_l0, y_train_l0, y_validation_l0, y_test_l0 = split_data(temp_l0)
                
                X_train_validation_params_l0, X_train_params_l0, X_validation_params_l0, X_test_params_l0, X_train_validation_ID_l0, X_train_ID_l0, X_validation_ID_l0, X_test_ID_l0 = get_params(X_train_validation_l0, X_train_l0, X_validation_l0, X_test_l0, temp_l0)
                
                X_train_validation_record_id = list(temp_l0['Record_ID'][X_train_validation_l0.index])
                X_train_record_id = list(temp_l0['Record_ID'][X_train_l0.index])
                X_validation_record_id = list(temp_l0['Record_ID'][X_validation_l0.index])
                X_test_record_id = list(temp_l0['Record_ID'][X_test_l0.index])
                y_train_validation_record_id = list(temp_l0['Record_ID'][y_train_validation_l0.index])
                y_train_record_id = list(temp_l0['Record_ID'][y_train_l0.index])
                y_validation_record_id = list(temp_l0['Record_ID'][y_validation_l0.index])
                y_test_record_id = list(temp_l0['Record_ID'][y_test_l0.index])                    

                X_train_validation_l1 = temp_l1[temp_l1['Record_ID'].isin(X_train_validation_record_id)]['X']   
                X_train_l1 = temp_l1[temp_l1['Record_ID'].isin(X_train_record_id)]['X']                    
                X_validation_l1 = temp_l1[temp_l1['Record_ID'].isin(X_validation_record_id)]['X']   
                X_test_l1 = temp_l1[temp_l1['Record_ID'].isin(X_test_record_id)]['X'] 
                X_train_validation_params_l1 = temp_l1[temp_l1['Record_ID'].isin(X_train_validation_record_id)]['ParameterID']   
                X_train_params_l1 = temp_l1[temp_l1['Record_ID'].isin(X_train_record_id)]['ParameterID']                    
                X_validation_params_l1 = temp_l1[temp_l1['Record_ID'].isin(X_validation_record_id)]['ParameterID']   
                X_test_params_l1 = temp_l1[temp_l1['Record_ID'].isin(X_test_record_id)]['ParameterID']                      
                X_train_validation_ID_l1 = temp_l1[temp_l1['Record_ID'].isin(X_train_validation_record_id)]['Record_ID']   
                X_train_ID_l1 = temp_l1[temp_l1['Record_ID'].isin(X_train_record_id)]['Record_ID']                    
                X_validation_ID_l1 = temp_l1[temp_l1['Record_ID'].isin(X_validation_record_id)]['Record_ID']   
                X_test_ID_l1 = temp_l1[temp_l1['Record_ID'].isin(X_test_record_id)]['Record_ID']                                          
                y_train_validation_l1 = temp_l1[temp_l1['Record_ID'].isin( y_train_validation_record_id)]['y']   
                y_train_l1 = temp_l1[temp_l1['Record_ID'].isin(y_train_record_id)]['y']   
                y_validation_l1 = temp_l1[temp_l1['Record_ID'].isin(y_validation_record_id)]['y']   
                y_test_l1 = temp_l1[temp_l1['Record_ID'].isin(y_test_record_id)]['y']     
                
                go_on = True
            else:
                X_train_l0 = 'N/A'
                X_train_validation_l0 = 'N/A'
                X_validation_l0 = 'N/A'
                X_test_l0 = 'N/A'
                X_train_validation_params_l0 = 'N/A'
                X_train_params_l0 = 'N/A'
                X_validation_params_l0 = 'N/A'
                X_test_params_l0 = 'N/A' 
                X_train_validation_ID_l0 = 'N/A'
                X_train_ID_l0 = 'N/A'
                X_validation_ID_l0 = 'N/A'
                X_test_ID_l0 = 'N/A' 
                y_train_l0 = 'N/A'
                y_train_validation_l0 = 'N/A'
                y_validation_l0 = 'N/A'
                y_test_l0 = 'N/A'
                
                X_train_l1 = 'N/A'
                X_train_validation_l1 = 'N/A'
                X_validation_l1 = 'N/A'
                X_test_l1 = 'N/A'
                X_train_validation_params_l1 = 'N/A'
                X_train_params_l1 = 'N/A'
                X_validation_params_l1 = 'N/A'
                X_test_params_l1 = 'N/A' 
                X_train_validation_ID_l1 = 'N/A'
                X_train_ID_l1 = 'N/A'
                X_validation_ID_l1 = 'N/A'
                X_test_ID_l1 = 'N/A' 
                y_train_l1 = 'N/A'
                y_train_validation_l1 = 'N/A'
                y_validation_l1 = 'N/A'
                y_test_l1 = 'N/A'                                       
                
                go_on = False
            
            output_temp_l0['X_train_validation'] = X_train_validation_l0
            output_temp_l0['X_train'] = X_train_l0
            output_temp_l0['X_validation'] = X_validation_l0
            output_temp_l0['X_test'] = X_test_l0
            output_temp_l0['X_train_validation_params'] = X_train_validation_params_l0
            output_temp_l0['X_train_params'] = X_train_params_l0
            output_temp_l0['X_validation_params'] = X_validation_params_l0
            output_temp_l0['X_test_params'] = X_test_params_l0 
            output_temp_l0['X_train_validation_ID'] = X_train_validation_ID_l0
            output_temp_l0['X_train_ID'] = X_train_ID_l0
            output_temp_l0['X_validation_ID'] = X_validation_ID_l0
            output_temp_l0['X_test_ID'] = X_test_ID_l0                 
            output_temp_l0['y_train_validation'] = y_train_validation_l0
            output_temp_l0['y_train'] = y_train_l0             
            output_temp_l0['y_validation'] = y_validation_l0
            output_temp_l0['y_test'] = y_test_l0
            
            output_temp_l1['X_train_validation'] = X_train_validation_l1
            output_temp_l1['X_train'] = X_train_l1
            output_temp_l1['X_validation'] = X_validation_l1
            output_temp_l1['X_test'] = X_test_l1
            output_temp_l1['X_train_validation_params'] = X_train_validation_params_l1
            output_temp_l1['X_train_params'] = X_train_params_l1
            output_temp_l1['X_validation_params'] = X_validation_params_l1
            output_temp_l1['X_test_params'] = X_test_params_l1 
            output_temp_l1['X_train_validation_ID'] = X_train_validation_ID_l1
            output_temp_l1['X_train_ID'] = X_train_ID_l1
            output_temp_l1['X_validation_ID'] = X_validation_ID_l1
            output_temp_l1['X_test_ID'] = X_test_ID_l1                 
            output_temp_l1['y_train_validation'] = y_train_validation_l1
            output_temp_l1['y_train'] = y_train_l1             
            output_temp_l1['y_validation'] = y_validation_l1
            output_temp_l1['y_test'] = y_test_l1  
            
            output[n] = output_temp_l0 
            output[n+1] = output_temp_l1
           
            info_temp_l0 = pd.DataFrame()
            info_temp_l0.loc[0,'n'] = n
            info_temp_l0.loc[0,'n_0'] = sum_n0_l0
            info_temp_l0.loc[0,'n_1'] = sum_n1_l0             
            info_temp_l0.loc[0,'data_option'] = 'V'            
            info_temp_l0.loc[0,'level'] = 'macro'
            info_temp_l0.loc[0,'column'] = column
            info_temp_l0.loc[0,'go_on'] = go_on
            
            info_temp_l1 = pd.DataFrame()
            info_temp_l1.loc[0,'n'] = n+1
            info_temp_l1.loc[0,'n_0'] = sum_n0_l1
            info_temp_l1.loc[0,'n_1'] = sum_n1_l1            
            info_temp_l1.loc[0,'data_option'] = 'V'            
            info_temp_l1.loc[0,'level'] = 'micro'
            info_temp_l1.loc[0,'column'] = column
            info_temp_l1.loc[0,'go_on'] = go_on 
  
            output_info = pd.concat([output_info, info_temp_l0], axis = 0, ignore_index  = True)
            output_info = pd.concat([output_info, info_temp_l1], axis = 0, ignore_index  = True)
            n += 2
   
    elif category == 'P':
        columns = [column for column in data.columns if bool(re.search('P[0-9]', column))]
        
        for column in columns:
            print('Processing sub-category: ', column)
            output_temp_l0 = {}
            output_temp_l1 = {}
            temp_l0 = macro_data.loc[:, ['Record_ID', 'ParameterID', column, 'ChartValue']].rename(columns = {column:'y', 'ChartValue':'X'})
            temp_l1 = data.loc[:, ['Record_ID', 'ParameterID',column, 'ChartValue']].rename(columns = {column:'y', 'ChartValue':'X'})
            sum_n1_l0 = temp_l0['y'].sum()
            sum_n0_l0 = len(temp_l0) - sum_n1_l0       
            
            sum_n1_l1 = temp_l1['y'].sum()
            sum_n0_l1 = len(temp_l1) - sum_n1_l1  
            
            # checking if there are at least 11 patients in each class 
            # just remembering that each line refers to a patient in sum_n1_l0
            if sum_n1_l0 >= 11:                
                X_train_validation_l0, X_train_l0, X_validation_l0, X_test_l0, y_train_validation_l0, y_train_l0, y_validation_l0, y_test_l0 = split_data(temp_l0)
                
                X_train_validation_params_l0, X_train_params_l0, X_validation_params_l0, X_test_params_l0, X_train_validation_ID_l0, X_train_ID_l0, X_validation_ID_l0, X_test_ID_l0 = get_params(X_train_validation_l0, X_train_l0, X_validation_l0, X_test_l0, temp_l0)                    
                
                X_train_validation_record_id = list(temp_l0['Record_ID'][X_train_validation_l0.index])
                X_train_record_id = list(temp_l0['Record_ID'][X_train_l0.index])
                X_validation_record_id = list(temp_l0['Record_ID'][X_validation_l0.index])
                X_test_record_id = list(temp_l0['Record_ID'][X_test_l0.index])
                y_train_validation_record_id = list(temp_l0['Record_ID'][y_train_validation_l0.index])
                y_train_record_id = list(temp_l0['Record_ID'][y_train_l0.index])
                y_validation_record_id = list(temp_l0['Record_ID'][y_validation_l0.index])
                y_test_record_id = list(temp_l0['Record_ID'][y_test_l0.index])
                
                X_train_validation_l1 = temp_l1[temp_l1['Record_ID'].isin(X_train_validation_record_id)]['X']   
                X_train_l1 = temp_l1[temp_l1['Record_ID'].isin(X_train_record_id)]['X']                    
                X_validation_l1 = temp_l1[temp_l1['Record_ID'].isin(X_validation_record_id)]['X']   
                X_test_l1 = temp_l1[temp_l1['Record_ID'].isin(X_test_record_id)]['X'] 
                X_train_validation_params_l1 = temp_l1[temp_l1['Record_ID'].isin(X_train_validation_record_id)]['ParameterID']   
                X_train_params_l1 = temp_l1[temp_l1['Record_ID'].isin(X_train_record_id)]['ParameterID']                    
                X_validation_params_l1 = temp_l1[temp_l1['Record_ID'].isin(X_validation_record_id)]['ParameterID']   
                X_test_params_l1 = temp_l1[temp_l1['Record_ID'].isin(X_test_record_id)]['ParameterID']                      
                X_train_validation_ID_l1 = temp_l1[temp_l1['Record_ID'].isin(X_train_validation_record_id)]['Record_ID']   
                X_train_ID_l1 = temp_l1[temp_l1['Record_ID'].isin(X_train_record_id)]['Record_ID']                    
                X_validation_ID_l1 = temp_l1[temp_l1['Record_ID'].isin(X_validation_record_id)]['Record_ID']   
                X_test_ID_l1 = temp_l1[temp_l1['Record_ID'].isin(X_test_record_id)]['Record_ID']                                          
                y_train_validation_l1 = temp_l1[temp_l1['Record_ID'].isin( y_train_validation_record_id)]['y']   
                y_train_l1 = temp_l1[temp_l1['Record_ID'].isin(y_train_record_id)]['y']   
                y_validation_l1 = temp_l1[temp_l1['Record_ID'].isin(y_validation_record_id)]['y']   
                y_test_l1 = temp_l1[temp_l1['Record_ID'].isin(y_test_record_id)]['y']    
                
                go_on = True
            else:
                X_train_l0 = 'N/A'
                X_train_validation_l0 = 'N/A'
                X_validation_l0 = 'N/A'
                X_test_l0 = 'N/A'
                X_train_validation_params_l0 = 'N/A'
                X_train_params_l0 = 'N/A'
                X_validation_params_l0 = 'N/A'
                X_test_params_l0 = 'N/A' 
                X_train_validation_ID_l0 = 'N/A'
                X_train_ID_l0 = 'N/A'
                X_validation_ID_l0 = 'N/A'
                X_test_ID_l0 = 'N/A' 
                y_train_l0 = 'N/A'
                y_train_validation_l0 = 'N/A'
                y_validation_l0 = 'N/A'
                y_test_l0 = 'N/A'
                
                X_train_l1 = 'N/A'
                X_train_validation_l1 = 'N/A'
                X_validation_l1 = 'N/A'
                X_test_l1 = 'N/A'
                X_train_validation_params_l1 = 'N/A'
                X_train_params_l1 = 'N/A'
                X_validation_params_l1 = 'N/A'
                X_test_params_l1 = 'N/A' 
                X_train_validation_ID_l1 = 'N/A'
                X_train_ID_l1 = 'N/A'
                X_validation_ID_l1 = 'N/A'
                X_test_ID_l1 = 'N/A' 
                y_train_l1 = 'N/A'
                y_train_validation_l1 = 'N/A'
                y_validation_l1 = 'N/A'
                y_test_l1 = 'N/A'                                                          
                
                go_on = False
            
            output_temp_l0['X_train_validation'] = X_train_validation_l0
            output_temp_l0['X_train'] = X_train_l0
            output_temp_l0['X_validation'] = X_validation_l0
            output_temp_l0['X_test'] = X_test_l0
            output_temp_l0['X_train_validation_params'] = X_train_validation_params_l0
            output_temp_l0['X_train_params'] = X_train_params_l0
            output_temp_l0['X_validation_params'] = X_validation_params_l0
            output_temp_l0['X_test_params'] = X_test_params_l0 
            output_temp_l0['X_train_validation_ID'] = X_train_validation_ID_l0
            output_temp_l0['X_train_ID'] = X_train_ID_l0
            output_temp_l0['X_validation_ID'] = X_validation_ID_l0
            output_temp_l0['X_test_ID'] = X_test_ID_l0                 
            output_temp_l0['y_train_validation'] = y_train_validation_l0
            output_temp_l0['y_train'] = y_train_l0             
            output_temp_l0['y_validation'] = y_validation_l0
            output_temp_l0['y_test'] = y_test_l0
            
            output_temp_l1['X_train_validation'] = X_train_validation_l1
            output_temp_l1['X_train'] = X_train_l1
            output_temp_l1['X_validation'] = X_validation_l1
            output_temp_l1['X_test'] = X_test_l1
            output_temp_l1['X_train_validation_params'] = X_train_validation_params_l1
            output_temp_l1['X_train_params'] = X_train_params_l1
            output_temp_l1['X_validation_params'] = X_validation_params_l1
            output_temp_l1['X_test_params'] = X_test_params_l1 
            output_temp_l1['X_train_validation_ID'] = X_train_validation_ID_l1
            output_temp_l1['X_train_ID'] = X_train_ID_l1
            output_temp_l1['X_validation_ID'] = X_validation_ID_l1
            output_temp_l1['X_test_ID'] = X_test_ID_l1                 
            output_temp_l1['y_train_validation'] = y_train_validation_l1
            output_temp_l1['y_train'] = y_train_l1             
            output_temp_l1['y_validation'] = y_validation_l1
            output_temp_l1['y_test'] = y_test_l1   
            
            output[n] = output_temp_l0
            output[n+1] = output_temp_l1

            info_temp_l0 = pd.DataFrame()
            info_temp_l0.loc[0,'n'] = n
            info_temp_l0.loc[0,'n_0'] = sum_n0_l0
            info_temp_l0.loc[0,'n_1'] = sum_n1_l0             
            info_temp_l0.loc[0,'data_option'] = 'P'            
            info_temp_l0.loc[0,'level'] = 'macro'
            info_temp_l0.loc[0,'column'] = column
            info_temp_l0.loc[0,'go_on'] = go_on
                            
            info_temp_l1 = pd.DataFrame()
            info_temp_l1.loc[0,'n'] = n+1
            info_temp_l1.loc[0,'n_0'] = sum_n0_l1
            info_temp_l1.loc[0,'n_1'] = sum_n1_l1            
            info_temp_l1.loc[0,'data_option'] = 'P'            
            info_temp_l1.loc[0,'level'] = 'micro'
            info_temp_l1.loc[0,'column'] = column
            info_temp_l1.loc[0,'go_on'] = go_on 
            
            output_info = pd.concat([output_info, info_temp_l0], axis = 0, ignore_index  = True)
            output_info = pd.concat([output_info, info_temp_l1], axis = 0, ignore_index  = True)
            n += 2
            
output_info.set_index('n', inplace = True)
            
output['info'] = output_info

with open(output_file, 'wb') as x:
    pickle.dump(output, x, protocol=pickle.HIGHEST_PROTOCOL)                             

    output_info.to_excel(output_summary) 
#------------------------------------------------------- building datasets ----


# Registering final time
b = time.time()        
print('--end--')
print('Total processing time: %0.2f minutos' %((b-a)/60))    
############################################################# Main routine ####