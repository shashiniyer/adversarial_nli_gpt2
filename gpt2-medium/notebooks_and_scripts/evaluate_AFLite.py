# imports
import argparse
from datasets import load_dataset, disable_caching, Dataset
from transformers import GPT2TokenizerFast, DataCollatorWithPadding, set_seed
import torch
from torch.nn.functional import one_hot
import numpy as np
import sys
sys.path.append('..')
from utils_ import tokenize, evaluate_acc_rk, train_classifier
import json
import pandas as pd
import pickle

# utility function to concatenate datasets and return 'datasets.Dataset' in torch format
def conc_prep_datasets(heuristic, key_list, stress_tests_datasets):
    
    # concat datasets
    out = stress_tests_datasets[key_list[0]]
    
    for key in key_list[1:]:
        
        out = pd.concat([out, stress_tests_datasets[key]], axis = 0)
    
    # one-hot encode labels
    out = Dataset.from_pandas(out).map(lambda x: \
        {'label': one_hot(torch.tensor(text_label_encoder[x['gold_label']]), 3).type(torch.float32).numpy()})
    
    # tokenize
    out = out.map(lambda x: tokenize(tokenizer, x['sentence1'] + '|' + x['sentence2']))
    len_bef_exclusion = len(out)
    
    # exclude instances with > 128 tokens
    out = out.filter(lambda x: x['exclude'] == False)
    len_aft_exclusion = len(out)

    # print message if instances were in fact excluded
    if len_bef_exclusion - len_aft_exclusion > 0:

        print(f'Heuristic: {heuristic} - {len_bef_exclusion - len_aft_exclusion} ' + \
              f'({(len_bef_exclusion/len_aft_exclusion - 1)*100:>2f}%) sequences excluded')
    
    # format data as torch tensors
    out.set_format(type = 'torch', columns = ['label', 'input_ids', 'attention_mask'])
    
    return(out)

# utility function to read in data and pre-process
def anli_data(split, tokenizer):

    # read in data
    data = load_dataset('anli', split = split)
    
    # one-hot encode labels
    data = data.map(lambda x: {'label': one_hot(torch.tensor(x['label']), 3).type(torch.float32).numpy()}, \
        batched = True)
    
    # tokenize data
    data = data.map(lambda x: tokenize(tokenizer, x['premise'] + '|' + x['hypothesis']))
    len_bef_exclusion = len(data)

    # exclude instances with > 128 tokens
    data = data.filter(lambda x: x['exclude'] == False)
    len_aft_exclusion = len(data)

    # print message if instances were in fact excluded
    if len_bef_exclusion - len_aft_exclusion > 0:

        print(f'Split: {split} - {len_bef_exclusion - len_aft_exclusion} ' + \
              f'({(len_bef_exclusion/len_aft_exclusion - 1)*100:>2f}%) sequences excluded')
    
    # keep only needed columns, set data format to PyTorch
    data.set_format(type = 'torch', columns = ['label', 'input_ids', 'attention_mask'])
    
    # store in data_dict
    return(data)

if __name__ == '__main__':

    ############# SETUP #############
    
    # get SMI input
    parser = argparse.ArgumentParser()
    parser.add_argument('model_seed')
    args = parser.parse_args()
    
    seed = int(args.model_seed)
    
    if seed not in [0, 1, 2, 3, 4]:
        
        seed = 0
        print('WARNING: Defaulting to Seed 0')
    
    # global settings
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    set_seed(42)
    disable_caching()
    
    # set up tokeniser
    # padding to left because GPT2 uses last token for prediction
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2-medium", padding_side = 'left', \
                                                  padding = True, truncation = True)
    tokenizer.pad_token = tokenizer.eos_token # pad with 'eos' token

    # set up data collator - https://huggingface.co/docs/transformers/main_classes/data_collator
    # this is a (callable) helper object that sends batches of data to the model
    data_collator = DataCollatorWithPadding(tokenizer, padding = 'max_length', \
                                             return_tensors = 'pt', max_length = 128)
    
    # load model
    model = torch.load(f'AFLite_fine_tuned_model_seed{seed}.pth')
    if model.training:
        model.eval()
    
    ############# In-Distribution Evaluation - SNLI test - Zero-Shot #############
    
    # log
    print('Begin: In-Distribution Evaluation - SNLI test - Zero-Shot')
    
    # read in data, remove data without gold-standard labels, one-hot encode labels
    snli_test = load_dataset('snli', split = 'test')
    snli_test = snli_test.filter(lambda x: x['label'] != -1).map( \
        lambda x: {'label': one_hot(torch.tensor(x['label']), 3).type(torch.float32).numpy()}, \
        batched = True)

    # tokenize data
    snli_test = snli_test.map(lambda x: tokenize(tokenizer, x['premise'] + '|' + x['hypothesis']))
    len_bef_exclusion = len(snli_test)

    # exclude instances with > 128 tokens
    snli_test = snli_test.filter(lambda x: x['exclude'] == False)
    len_aft_exclusion = len(snli_test)

    # print message if instances were in fact excluded
    if len_bef_exclusion - len_aft_exclusion > 0:

        print(f'{len_bef_exclusion - len_aft_exclusion} ' + \
              f'({(len_bef_exclusion/len_aft_exclusion - 1)*100:>2f}%) sequences excluded')

    # format data as torch tensors
    snli_test.set_format(type = 'torch', columns = ['label', 'input_ids', 'attention_mask'])
    
    # set up dataloader (batch generator)
    dataloader = torch.utils.data.DataLoader(snli_test, batch_size=128, collate_fn=data_collator)
    
    # evaluate model
    acc, rk = evaluate_acc_rk(model, dataloader, device)
    print(f'Seed: {seed} - Dataset: SNLI (test) - Accuracy: {acc*100:>3f}%, RK: {rk:>3f}')
    
    # free up some RAM
    del snli_test

    # log
    print('End: In-Distribution Evaluation - SNLI test - Zero-Shot\n')
    
    ############# Out-of-Distribution Evaluation - HANS - Zero-Shot #############
    
    # log
    print('Begin: Out-of-Distribution Evaluation - HANS - Zero-Shot')
    
    # read in data, remove data without gold-standard labels, one-hot encode labels
    hans = load_dataset('hans', split = 'validation')
    hans = hans.filter(lambda x: x['label'] != -1).map( \
        lambda x: {'label': one_hot(torch.tensor(x['label']), 3).type(torch.float32).numpy()}, \
        batched = True)    
    
    # tokenize data
    hans = hans.map(lambda x: tokenize(tokenizer, x['premise'] + '|' + x['hypothesis']))
    len_bef_exclusion = len(hans)

    # exclude instances with > 128 tokens
    hans = hans.filter(lambda x: x['exclude'] == False)
    len_aft_exclusion = len(hans)

    # print message if instances were in fact excluded
    if len_bef_exclusion - len_aft_exclusion > 0:

        print(f'{len_bef_exclusion - len_aft_exclusion} ' + \
              f'({(len_bef_exclusion/len_aft_exclusion - 1)*100:>2f}%) sequences excluded')

    # partition data by `heuristic` 
    data_dict = {x: hans.filter(lambda y: y['heuristic'] == x) \
                for x in ['constituent', 'lexical_overlap', 'subsequence']}

    # format as torch tensors
    for val in data_dict.values():

        val.set_format(type = 'torch', columns = ['label', 'input_ids', 'attention_mask'])
    
    # evaluate model
    for data_name, data in data_dict.items():

        # set up dataloader (batch generator)
        dataloader = torch.utils.data.DataLoader(data, batch_size=128, collate_fn=data_collator)

        # evaluate model
        acc, rk = evaluate_acc_rk(model, dataloader, device, problem = 'TE')
        print(f'Seed: {seed} - Dataset: {data_name} - Accuracy: {acc*100:>3f}%, RK: {rk:>3f}')

    # free up some RAM
    del hans
    del data_dict    
    
    # log
    print('End: Out-of-Distribution Evaluation - HANS - Zero-Shot')
    
    ############# Out-of-Distribution Evaluation - NLI Diagnostics - Zero-Shot #############
    
    # log
    print('Begin: Out-of-Distribution Evaluation - NLI Diagnostics - Zero-Shot')
    
    # read in data, encode gold-standard labels from str->numeric, one-hot encode numeric labels
    nli_diag = Dataset.from_pandas(pd.read_csv('../raw_data/diagnostic-full.tsv', delimiter = '\t'))
    text_label_encoder = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
    nli_diag = nli_diag.map( \
        lambda x: {'label': one_hot(torch.tensor(text_label_encoder[x['Label']]), 3).type(torch.float32).numpy()})

    # tokenize data
    nli_diag = nli_diag.map(lambda x: tokenize(tokenizer, x['Premise'] + '|' + x['Hypothesis']))
    len_bef_exclusion = len(nli_diag)

    # exclude instances with > 128 tokens
    nli_diag = nli_diag.filter(lambda x: x['exclude'] == False)
    len_aft_exclusion = len(nli_diag)

    # print message if instances were in fact excluded
    if len_bef_exclusion - len_aft_exclusion > 0:

        print(f'{len_bef_exclusion - len_aft_exclusion} ' + \
              f'({(len_bef_exclusion/len_aft_exclusion - 1)*100:>2f}%) sequences excluded')

    # partition data by heuristic
    data_dict = {x: nli_diag.filter(lambda y: y[x] is not None) \
                for x in ['Lexical Semantics', 'Predicate-Argument Structure', 'Logic', 'Knowledge']}

    # format as torch tensors
    for val in data_dict.values():

        val.set_format(type = 'torch', columns = ['label', 'input_ids', 'attention_mask'])    
    
    # evaluate model
    for data_name, data in data_dict.items():

        # set up dataloader (batch generator)
        dataloader = torch.utils.data.DataLoader(data, batch_size=128, collate_fn=data_collator)

        # evaluate model
        acc, rk = evaluate_acc_rk(model, dataloader, device)
        print(f'Seed: {seed} - Dataset: {data_name} - Accuracy: {acc*100:>3f}%, RK: {rk:>3f}')

    # free up some RAM
    del nli_diag
    del data_dict

    # log
    print('End: Out-of-Distribution Evaluation - NLI Diagnostics - Zero-Shot')
    
    ############# Out-of-Distribution Evaluation - Stress Tests - Zero-Shot #############
    
    # log
    print('Begin: Out-of-Distribution Evaluation - Stress Tests - Zero-Shot')
    
    # read in prepared stress_test_datasets
    with open('../stress_tests_datasets.pkl', 'rb') as f:
        stress_tests_datasets = pickle.load(f)    
    
    # partition data by heuristic + pre-process them
    data_dict = {'Competence': conc_prep_datasets('Competence', \
                                 ['antonym_matched', 'antonym_mismatched', 'quant_hard'], stress_tests_datasets), \
                 'Distraction': conc_prep_datasets('Distraction', \
                                 ['taut2_matched', 'taut2_mismatched', 'negation_matched', 'negation_mismatched', \
                                 'length_mismatch_matched', 'length_mismatch_mismatched'], stress_tests_datasets), \
                 'Noise': conc_prep_datasets('Noise', \
                                 [k for k in stress_tests_datasets.keys() if k.startswith('dev_gram')], stress_tests_datasets)}
    
    # evaluate model
    for data_name, data in data_dict.items():

        # set up dataloader (batch generator)
        dataloader = torch.utils.data.DataLoader(data, batch_size=128, collate_fn=data_collator)

        # evaluate model
        acc, rk = evaluate_acc_rk(model, dataloader, device)
        print(f'Seed: {seed} - Dataset: {data_name} - Accuracy: {acc*100:>3f}%, RK: {rk:>3f}')

    # free up some RAM
    del stress_tests_datasets
    del data_dict
    del model
    
    # log
    print('End: Out-of-Distribution Evaluation - Stress Tests - Zero-Shot')
    
    ############# Out-of-Distribution Evaluation - ANLI - Fine-Tuning #############
    
    # log
    print('Begin: Out-of-Distribution Evaluation - ANLI - Fine-Tuning')    
    
    # fine-tune and evaluate for each round
    for rd in ['r1', 'r2', 'r3']:
        
        # fine-tune
        tr = anli_data('train_' + rd, tokenizer)
        tr_model = torch.load(f'AFLite_fine_tuned_model_seed{seed}.pth') # load each time to reduce RAM load
        tr_dataloader = torch.utils.data.DataLoader(tr, batch_size=32, shuffle=True, collate_fn=data_collator)
        optimizer = torch.optim.Adam(tr_model.parameters(), lr = 1e-5)
        trained_classifier = train_classifier(tr_model, tr_dataloader, optimizer, device)
        del tr
        
        # evaluate
        te = anli_data('test_' + rd, tokenizer)
        te_dataloader = torch.utils.data.DataLoader(te, batch_size=32, collate_fn=data_collator)
        acc, rk = evaluate_acc_rk(trained_classifier, te_dataloader, device)
        del te
        del tr_model
        print(f'Round: {rd} - Seed: {seed} - Accuracy: {acc*100:>3f}%, RK: {rk:>3f}')           
    
    # log
    print('End: Out-of-Distribution Evaluation - ANLI - Fine-Tuning')     