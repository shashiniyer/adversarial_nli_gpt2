import os
import pickle
import json
import pandas as pd

# load in files from './raw_data/Stress Tests/'
stress_tests_datasets = {}  
count = 0

for path, subdirs, files in os.walk('./raw_data/Stress Tests/'):

    for name in files:
        
        if name.endswith('.jsonl'):
            
            count += 1
            
            dict_key = name.replace('multinli_0.9_', '').replace('.jsonl', '')
            
            print('File ' + str(count) + ': ' + dict_key + ' - Begin')
        
            with open(os.path.join(path, name), 'r') as json_file:
                
                json_list = list(json_file)
            
            if list in [type(x) for x in json.loads(json_list[0]).values()]:
            
                df = pd.DataFrame(json.loads(json_list[0]))
            
            else:
                
                df = pd.DataFrame(json.loads(json_list[0]), index = [0])
            
            for i in range(1, len(json_list)):
                
                if list in [type(x) for x in json.loads(json_list[i]).values()]:
                
                    df = pd.concat([df, pd.DataFrame(json.loads(json_list[i]))], axis = 0)
                
                else:
                    
                    df = pd.concat([df, pd.DataFrame(json.loads(json_list[i]), index = [0])], axis = 0)
            
            stress_tests_datasets[dict_key] = df.reset_index()
            print('File ' + str(count) + ': ' + dict_key + ' - End')

# retain only necessary columns
for k, v in stress_tests_datasets.items():
    
    stress_tests_datasets[k] = v.loc[:, ['gold_label', 'sentence1', 'sentence2']]
    
# write out stress_tests_datasets dictionary
with open('stress_tests_datasets.pkl', 'wb') as f:
   
    pickle.dump(stress_tests_datasets, f)