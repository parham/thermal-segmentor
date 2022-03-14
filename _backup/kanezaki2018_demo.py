
from argparse import ArgumentParser
from pytorch_lightning import Trainer
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from phm2.core import UnsupervisedDataset, load_config
from phm2.kanezaki2018 import Kanezaki2018Module, Kanezaki2018Loop


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    config = load_config('kanezaki2018_config.json',dotflag=True)
    model = Kanezaki2018Module(3, config)

    dataset = UnsupervisedDataset('/home/phm/Development/thermal-segmentor/datasets')
    # dataset = ImageFolder('/home/phm/Development/thermal-segmentor/datasets')
    dataloader = DataLoader(dataset,batch_size=1, shuffle=True)
    
    # Example : python main.py --gpus 2 --max_steps 10 --limit_train_batches 10 --any_trainer_arg x
    trainer = Trainer.from_argparse_args(args, gpus=1)
    # trainer.fit_loop = Kanezaki2018Loop(model,
    #     optimizer=model.optimizers,
    #     config=config,
    #     dataloader=dataloader
    # )

    trainer.fit(model, train_dataloaders=dataloader)