###############################################################################
#                                                                             #
#                          machine learning approach                          #
#                           combining parts 1 and 2                           #
#                                                                June 23 2020 #
###############################################################################


### Loading libraries #########################################################
import time
import numpy as np
import pandas as pd
######################################################## Loading libraries ####


### Declaring I/O variables ###################################################
input_part1 = 'ML_summary_part1.pickle'
input_part2 = 'ML_summary_part2.pickle'
output = 'ML_reults_summary.xlsx'
################################################## Declaring I/O variables ####


### Main routine ##############################################################
# Registering initial time
a = time.time()    
print("--start--")

# Reading parts 1 and 2
part1 = pd.read_pickle(input_part1)
part2 = pd.read_pickle(input_part2)

if len(part1) > len(part2):
    raise ValueError('Part 1 > Part 2')    
elif len(part1) < len(part2):
    raise ValueError('Part 2 > Part 1')
    
# Initializing ouput
output = pd.DataFrame()    
    
# Combining parts 1 and 2
for n in range(1, len(part1) + 1):
    try:
        auc_val_p1 = float(part1.loc[n,'AUC Validation (95% CI)'][:5])
        auc_val_p2 = float(part2.loc[n,'AUC Validation (95% CI)'][:5])
        if auc_val_p1 >= auc_val_p2:
            output = pd.concat([output, part1.loc[[n]]], axis = 0)
        else:
            output = pd.concat([output, part2.loc[[n]]], axis = 0)         
    except:
        output = pd.concat([output,part1.loc[[n]]], axis = 0)
            
output.set_index('n', inplace = True)

# Saving to xslx
output.to_excel(output)

# Registering final time
b = time.time()        
print('--end--')
print('Total processing time: %0.2f minutos' %((b-a)/60))    
############################################################# Main routine ####