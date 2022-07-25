import torch
from tqdm.notebook import tqdm

def train_classifier(classifier, dataloader, optimizer, device, npochs = 3):
    
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
    for _ in range(npochs):
        
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

def predict(classifier, dataloader, device):
    
    classifier.to(device) # move classifier to device
    
    for batch, data in tqdm(enumerate(dataloader), total = len(dataloader)):
        
        with torch.no_grad():
            batch_preds = classifier(**data.to(device)).logits.argmax(1)
        
        #print(batch_preds)
        
        if batch == 0:
            
            preds = batch_preds.detach().cpu()
        
        else:
            
            preds = torch.cat((preds, batch_preds.detach().cpu()))
    
    classifier.to('cpu') # move classifier to cpu
    
    return(preds.numpy())

def tokenize(tokenizer, sequence):
    
    raw = tokenizer(sequence)
    
    if len(raw['input_ids']) > 128:
        
        # set Exclude to True
        return({'input_ids': raw['input_ids'], 'attention_mask': raw['attention_mask'], 'exclude': True})
    
    else:
        
        # set Exclude to False
        return({'input_ids': raw['input_ids'], 'attention_mask': raw['attention_mask'], 'exclude': False})