###############################################################################
#                                                                             #
#                          webscraper for names                               #
#                                                                June 23 2020 #
###############################################################################

### Loading libraries #########################################################
import urllib3
from bs4 import BeautifulSoup 
import re
import pandas as pd
import string
import pickle
######################################################## Loading libraries ####


### Declaring I/O variables ###################################################
url1 = "https://nameberry.com/search"
url2 = "https://babynames.net/names?page="
output = "names_dict.pickle"
################################################## Declaring I/O variables ####


### Declaring Functions #######################################################
def open_url(url):
    http = urllib3.PoolManager()
    try:
        bsObj = BeautifulSoup(http.request('GET', url).data, "lxml")
    except Exception as e:
        print(e)
    return bsObj
###################################################### Declaring Functions ####


### Main routine ##############################################################
# URL #1 (https://nameberry.com/search)

# site structure
#                     url1
#               /       |          \
#     girls_name    boys_names     unisex_names
#         |             |               |
#     A,B,...,Z     A,B,...,Z       A,B,...,Z

# Declaring site branches
links = ["girls_names", "boys_names", "unisex_names"]
letters = string.ascii_uppercase

# output list
names_list = []

# Accessing each of the branches 
for link in links:
    
    # Identifying gender based on link
    if link == 'girls_names':
        gender = 'F'
    elif link == 'boys_names':
        gender = 'M'
    else:
        gender = 'U'
        
    for letter in letters:
        print("Processing: {}/{}/{}".format(url1,link,letter))
        
        # Opening main page of the branch to identify the number of sub-pages
        main_html = open_url(url1+"/"+link+"/"+letter)
        try:
            n_pages = int(re.findall('[0-9]?[0-9]',
                                      main_html.findAll('span', 
                                                        {'class':'last'})[0].find('a').get_attribute_list('href')[0])[0])
        except:
            n_pages = 1
                
        # Opening each of the sub-pages of the branch
        for page in range(1,n_pages):
            html = open_url(url1+"/"+link+"/"+letter+'?page='+str(page))
            
            # Slicing HTML to extract the part that contains names
            items = html.findAll('h4')
            
            # Saving names
            for item in items:
                name = item.get_text().strip()
                names_list.append((name, gender))

# Creating dataframe with data from url1            
df_url1 = pd.DataFrame(names_list, columns = ['Name', 'Gender'])

#-----------------------------------------------------------------------------
# URL #2 ("https://babynames.net/names?page=)

# site structure

#             url2
#              |
#         1, 2, ..., 727

# Names are presented in a list that includes all letters and genders

# output list
names_list = []

# Opening each of the branches 
for link in range(1,728):     
    print("Processing: {}{}".format(url2,link))    
    html = open_url(url2 + str(link))
    
    # Slicing HTML to extract parts that contains names and genders
    names = html.findAll(class_ ='result-name')
    genders = html.findAll(class_ = 'result-gender') 
    
    if len(names) != len(genders):
        print('len(names) != len(genders) at link: {}'.format(url2 + str(link)))
        continue
    else:
        for item in range(len(names)):
            name = names[item].text
            
            if 'boygirl' in str(genders[item]):
                gender = 'U'
            elif 'girl' in str(genders[item]):
                gender = 'F'
            else:
                gender = 'M'            
            names_list.append((name, gender))           
    
# Creating dataframe with data from url2         
df_url2 = pd.DataFrame(names_list, columns = ['Name', 'Gender'])

#-----------------------------------------------------------------------------
# Combining the two sources    
df_output = pd.concat([df_url1,df_url2], axis = 0)

# Checking for repeated names and adjusting gender
df_output = df_output.groupby('Name')['Gender'].apply(','.join).reset_index()
df_output['Gender'] = df_output['Gender'].apply(lambda x: 
                                                
                                                'M'
                                                if
                                                x == 'M' or
                                                x == 'M,M' 
                                                else 
                                                
                                                'F'
                                                if
                                                x == 'F' or
                                                x == 'F,F' 
                                                else 
                                                
                                                'U')

# Transforming into dict
dict_output = df_output.set_index(['Name']).to_dict()['Gender']

# Saving to pickle
with open(output, 'wb') as handle:
    pickle.dump(dict_output, handle, protocol=pickle.HIGHEST_PROTOCOL)
############################################################# Main routine ####
