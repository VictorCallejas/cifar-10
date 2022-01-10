import torch
import torch.nn as nn 

from tqdm import tqdm


def train_epoch(model, dataloader, optimizer, device, criterion, run, scaler, fp16 = False,swa=''):

    model.train()
    optimizer.zero_grad(set_to_none=True)

    labels, preds = torch.tensor([]), torch.tensor([])

    for step, batch in tqdm(enumerate(dataloader),total=len(dataloader),desc='TRAINING',leave=False):

        x, targets = batch
        x = x.to(device,non_blocking=True,dtype=torch.float32)
        targets = targets.to(device,non_blocking=True,dtype=torch.int64)
        
        with torch.cuda.amp.autocast(enabled=fp16):
            b_preds = nn.Softmax()(model(x))
            loss = criterion(b_preds, targets)

        scaler.scale(loss).backward()
        run[swa+"train/batch_loss"].log(loss.item())

        #scaler.unscale_(optimizer)
        #nn.utils.clip_grad_norm_(model.parameters(),2.0)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        preds = torch.cat([preds, b_preds.detach().cpu()], dim = 0)
        labels = torch.cat([labels, targets.detach().cpu()], dim = 0)   

    epoch_loss = criterion(preds,labels.long())
    _, arg_pred = preds.max(1)
    correct = arg_pred.eq(labels).sum().item()
    acc = correct/labels.shape[0]
    
    return epoch_loss.item(), acc



def valid_epoch(model, dataloader, device, criterion, run, scaler, fp16 = False,swa=''):

    model.eval()

    labels, preds = torch.tensor([]), torch.tensor([])

    for _, batch in tqdm(enumerate(dataloader),total=len(dataloader),desc='VALIDATION',leave=False):

        x, targets = batch
        x = x.to(device,non_blocking=True,dtype=torch.float32)
        targets = targets.to(device,non_blocking=True,dtype=torch.int64)
        
        with torch.cuda.amp.autocast(enabled=fp16):
            with torch.no_grad(): 
                b_preds = nn.Softmax()(model(x))
                loss = criterion(b_preds, targets)
        
                scaler.scale(loss)
                
                run[swa+"dev/batch_loss"].log(loss.item())

                preds = torch.cat([preds, b_preds.detach().cpu()], dim = 0)
                labels = torch.cat([labels, targets.detach().cpu()], dim = 0)
                 

    epoch_loss = criterion(preds,labels.long())
    _, arg_pred = preds.max(1)
    correct = arg_pred.eq(labels).sum().item()
    acc = correct/labels.shape[0]
    
    return epoch_loss.item(), acc


# MAKE MEDIAN OF TEST AUG
def test_epoch(model, dataloader, device, criterion, run, scaler, fp16 = False,swa=''):

    model.eval()

    labels, preds = torch.tensor([]), torch.tensor([])

    for _, batch in tqdm(enumerate(dataloader),total=len(dataloader),desc='TESTING',leave=False):

        x, targets = batch
        x = x.to(device,non_blocking=True,dtype=torch.float32)
        targets = targets.to(device,non_blocking=True,dtype=torch.int64)
        
        with torch.cuda.amp.autocast(enabled=fp16):
            with torch.no_grad():
                logits = model(x)
                b_preds = nn.Softmax()(logits)
                logits = torch.mean(logits, dim=1) # Augmentations on test time
                loss = criterion(b_preds, targets)
        
                scaler.scale(loss)
                
                run[swa+"test/batch_loss"].log(loss.item())

                preds = torch.cat([preds, b_preds.detach().cpu()], dim = 0)
                labels = torch.cat([labels, targets.detach().cpu()], dim = 0)
                 

    epoch_loss = criterion(preds,labels.long())
    _, arg_pred = preds.max(1)
    correct = arg_pred.eq(labels).sum().item()
    acc = correct/labels.shape[0]
    
    return epoch_loss.item(), acc


def save_ckpt(model, optimizer, dataloader, cfg, path):
    torch.save({
            'model':model.state_dict(),
            'model_name':cfg['model']['model_name'],
            'optimizer_name':cfg['train']['optimizer'],
            'optimizer':optimizer.state_dict(),
            'data':{
                'channels':dataloader.dataset.channels,
                'norm':{
                    'mean':0,#dataloader.dataset.t_mean,
                    'std':1,#dataloader.dataset.t_std
                }
            }
        },
        path
        )