import torch
from tqdm.notebook import tqdm
import numpy as np
from torchmetrics.functional import accuracy, matthews_corrcoef

def tokenize(tokenizer, sequence, max_model_length = 128):
    
    # apply tokenizer on sequence
    raw = tokenizer(sequence)
    
    if len(raw['input_ids']) > max_model_length:
        
        # set Exclude to True
        return({'input_ids': raw['input_ids'], 'attention_mask': raw['attention_mask'], 'exclude': True})
    
    else:
        
        # set Exclude to False
        return({'input_ids': raw['input_ids'], 'attention_mask': raw['attention_mask'], 'exclude': False})

def train_classifier(classifier, dataloader, optimizer, device, nepochs = 3):
    
    # initialise
    curr_loss = 0
    #prev_loss = float("inf")
    
    # move classifier to device and set it in train mode
    classifier.to(device)
    classifier.train()
    
    # cache training data size
    train_data_size = len(dataloader.dataset)
    
    # train until convergence
    # while abs(prev_loss - curr_loss) > 1e-5:
    
    # train for nepochs; nepochs = 3 in Le et al. (2020) - https://arxiv.org/abs/2002.04108
    for _ in range(nepochs):
        
        # reset losses
        #prev_loss = curr_loss
        curr_loss = 0
        
        for batch, data in tqdm(enumerate(dataloader), total = len(dataloader)):

            # Torch requirement
            classifier.zero_grad()

            # Compute prediction and loss
            outputs = classifier(**data.to(device))
            batch_loss = outputs[0]

            # Backpropagation
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
            optimizer.step()
            
            # Log
            if batch % int(len(dataloader)/10) == 0:
                batch_loss, current = batch_loss.item(), batch * len(data['labels'])
                print(f"loss: {batch_loss:>7f}  [{current:>5d}/{train_data_size:>5d}]")
            
            # Add up batch-loss
            curr_loss += batch_loss * len(data['labels'])
        
        # Average out curr_loss
        curr_loss /= train_data_size
        print(f'Epoch average loss: {curr_loss}')
        
    # set classifier to eval model and move it to CPU
    classifier.eval()
    classifier.to('cpu')

    # print done
    print('Done!')
    
    return(classifier)

def predict(classifier, dataloader, device, out_format = 'np', return_labels = False, problem = 'NLI'):
    
    assert (type(out_format) == str) and (out_format in ['np', 'pt-gpu']), \
        "Implementation available only for out_format in ['np', 'pt-gpu']"
    
    assert (type(problem) == str) and (problem in ['NLI', 'TE']), \
        "Implementation available only for problem in ['NLI', 'TE']"
    
    classifier.to(device) # move classifier to device
    
    if out_format == 'np':
    
        for batch, data in tqdm(enumerate(dataloader), total = len(dataloader)):

            with torch.no_grad():
                batch_preds = classifier(**data.to(device)).logits.argmax(1)

            if batch == 0:

                preds = batch_preds.detach().cpu()
                
                if return_labels:
                    
                    labels = data['labels'].argmax(1).detach().cpu()

            else:

                preds = torch.cat((preds, batch_preds.detach().cpu()))
                
                if return_labels:
                    
                    labels = torch.cat((labels, data['labels'].argmax(1).detach().cpu()))

        classifier.to('cpu') # move classifier to cpu
        
        if problem == 'TE':
            
            preds[preds == 2] = torch.tensor([1]).detach()
        
        if return_labels:
            
            return(preds.numpy(), labels.numpy())
        
        else:
            
            return(preds.numpy())
    
    elif out_format == 'pt-gpu':
    
        for batch, data in tqdm(enumerate(dataloader), total = len(dataloader)):

            with torch.no_grad():
                batch_preds = classifier(**data.to(device)).logits.argmax(1)

            #print(batch_preds)

            if batch == 0:

                preds = batch_preds.detach()
                
                if return_labels:
                    
                    labels = data['labels'].argmax(1).detach()

            else:

                preds = torch.cat((preds, batch_preds.detach()))
                
                if return_labels:
                    
                    labels = torch.cat((labels, data['labels'].argmax(1).detach()))

        classifier.to('cpu') # move classifier to cpu
        
        if problem == 'TE':
            
            preds[preds == 2] = torch.tensor([1]).detach()
        
        if return_labels:
            
            return(preds, labels)
        
        else:
            
            return(preds)

def select_k(pred_scores, tau, k, seed = 42):
    
    """
        Select up to k instances with the highest predictability
        scores (see report) subject to score >= tau
    """
    
    # shuffle to randomly resolve ties
    shuf_idx = np.arange(len(pred_scores)) # initialise
    np.random.default_rng(seed).shuffle(shuf_idx) #O(n) complexity
    shuf_scores = np.array(pred_scores)[shuf_idx] #O(n) complexity
    
    # sort descending because we want to select instances with high pred_score
    order_idx = np.argsort(-shuf_scores) #O(n²) complexity - most expensive operation (worst case)
    sort_shuf_idx = shuf_idx[order_idx] #O(n) complexity
    sort_shuf_scores = shuf_scores[order_idx] #O(n) complexity
    
    return(sort_shuf_idx[sort_shuf_scores >= tau][:k]) #O(n) complexity

def evaluate_acc_rk(classifier, dataloader, device, problem = 'NLI'):
    
    assert (type(problem) == str) and (problem in ['NLI', 'TE']), \
        "Implementation available only for problem in ['NLI', 'TE']"
    
    # get predictions and labels
    preds, labels = predict(classifier, dataloader, device, out_format = 'pt-gpu', return_labels = True, problem = problem)
    
    # set num_classes
    if problem == 'TE':
    
        num_classes = 2
    
    else:
        
        num_classes = 3
    
    return(accuracy(preds, labels).detach().cpu().numpy(), \
           matthews_corrcoef(preds, labels, num_classes = num_classes).detach().cpu().numpy())