import numpy as np 

from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from data.utils import unpickle
from data.dataset import CIFAR

# http://www.cs.toronto.edu/~kriz/cifar.html
def get_data(cfg):

    #meta = unpickle(cfg['data_path']+'batches.meta')
    #named_labels = meta[b'label_names']

    data = np.empty((0,3,32,32),dtype=np.uint8,)
    labels = np.array([],dtype=np.uint8)
    
    for i in range(1,5):
        batch = unpickle(cfg['dataset']['data_path']+'data_batch_'+str(i))
        data = np.concatenate([data,batch[b'data'].reshape((10000,3,32,32))], axis = 0)
        labels = np.concatenate([labels,batch[b'labels']], axis = 0)


    print(data.shape)
    print(labels.shape)

    return data, labels

def get_test_data(cfg):

    batch = unpickle(cfg['dataset']['data_path']+'test_batch')
    data = batch[b'data'].reshape((10000,3,32,32)).astype(np.uint8)
    labels = np.array(batch[b'labels'],dtype=np.uint8)

    print(data.shape)
    print(labels.shape)

    return data, labels



def get_dataloaders(cfg):

    x, y = get_data(cfg)
    x_test, y_test = get_test_data(cfg)

    if cfg['run']['fast_run']:
        x, y = x[:30], y[:30]

    #x_train, x_val, y_train, y_val = x, x[:10], y, y[:10] # Just train on all data
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=cfg['dataset']['val_size'], random_state=cfg['run']['SEED'])

    train_dataset = CIFAR(x_train, y_train, cfg, True)
    val_dataset = CIFAR(x_val, y_val, cfg, False)
    test_dataset = CIFAR(x_test, y_test, cfg, False)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=cfg['dataset']['batch_size'],
        num_workers=cfg['dataset']['num_workers'],
        pin_memory=True
    )

    val_dataloader = DataLoader(
        dataset=val_dataset,
        sampler=SequentialSampler(val_dataset),
        batch_size=cfg['dataset']['batch_size'],
        num_workers=cfg['dataset']['num_workers'],
        pin_memory=True
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=1,
        num_workers=cfg['dataset']['num_workers'],
        pin_memory=True
    )

    return train_dataloader, val_dataloader, test_dataloader