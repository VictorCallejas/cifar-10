
import torch

from training.utils import train_epoch, valid_epoch, save_ckpt, test_epoch

from tqdm import tqdm 

from models.cnn import CNN
from models.transformer import VIT


def train(train_dataloader, val_dataloader, device, run, cfg):


    if cfg['model']['class'] == 'cnn':
        model = CNN()
    elif cfg['model']['class'] == 'vit':
        model = VIT()
    else:
        raise ValueError('Model from cfg not found')
    
    print(model)
    model = model.to(device)
   

    optimizer = getattr(torch.optim,cfg['train']['optimizer'])(
                model.parameters(),
                lr = float(cfg['train']['lr'])
            )

    criterion = torch.nn.CrossEntropyLoss()

    fp16 = cfg['train']['fp16']
    scaler = torch.cuda.amp.GradScaler()

    path = cfg['train']['save_path'] + cfg['train']['save_name']

    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 160], gamma=1.0)

    for epoch in tqdm(range(0, cfg['train']['epochs']), total = cfg['train']['epochs'],desc='EPOCH',leave=False):
        t_loss, t_acc = train_epoch(model, train_dataloader, optimizer, device, criterion, run, scaler, fp16)
        run["train/loss"].log(t_loss)
        run["train/acc"].log(t_acc)

        v_loss, v_acc = valid_epoch(model, val_dataloader, device, criterion, run, scaler, fp16)
        run["dev/loss"].log(v_loss)
        run["dev/acc"].log(v_acc)

        print('TRAIN: ',t_loss, t_acc, 'VAL: ',v_loss, v_acc)

        for param_group in optimizer.param_groups:
            run["lr"].log(param_group['lr'])

        save_ckpt(model, optimizer, train_dataloader, cfg, path)

        #scheduler.step()

        run['model_pt'].track_files(path)
        


def test(dataloader, device, run, cfg):

    if cfg['model']['class'] == 'cnn':
        model = CNN()
    elif cfg['model']['class'] == 'vit':
        model = VIT()
    else:
        raise ValueError('Model from cfg not found')
    
    ckpt = torch.load(cfg['train']['save_path'] + cfg['train']['save_name'])
    model.load_state_dict(ckpt['model'])
    model = model.to(device)

    fp16 = cfg['train']['fp16']
    scaler = torch.cuda.amp.GradScaler()
    criterion = torch.nn.CrossEntropyLoss()

    test_loss, test_acc = test_epoch(model, dataloader, device, criterion, run, scaler, fp16)
    run["test/loss"].log(test_loss)
    run["test/acc"].log(test_acc)

    print('TEST: ',test_loss, test_acc)

    
        