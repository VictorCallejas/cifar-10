from importlib.util import module_for_loader
from omegaconf import DictConfig, OmegaConf

from utils.utils import config_logs, set_deterministic, get_device
from data.data import get_dataloaders
from training.train import train, test

import warnings
warnings.filterwarnings("ignore")

def main(cfg : DictConfig) -> None:

    print(OmegaConf.to_yaml(cfg))

    #INIT RUN
    run = config_logs()
    run['cfg'] = cfg

    set_deterministic(cfg['run']['SEED'])
    device = get_device(cfg['train']['device'])

    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(cfg)
    try:
        train(train_dataloader, val_dataloader, device, run, cfg)
    except KeyboardInterrupt as err:
        print('Training interrupted, starting testing')

    test(test_dataloader, device, run, cfg)
    
    print('--- PROGRAM END ---')


if __name__ == "__main__":
    cfg = OmegaConf.load('config/cfg.yaml')
    main(cfg)