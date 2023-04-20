import torch
import importlib
import wandb


def run_model_pipeline(config):
    if config['mode'] == 'train':

        # Get the model trainer
        trainer_path = "trainers." + 'trainer_' + config['model_name']
        trainer = importlib.import_module(trainer_path)

        def train():
            with wandb.init(mode='disabled') as run:
                # Initialize the model, data and optimization problem
                model, train_loader, valid_loader, criterion, optimizer = trainer.make(config)

                # Train the model
                trainer.train(model, train_loader, valid_loader, criterion, optimizer, config, save_model=True)

        train()

    elif config['mode'] == 'infer':
        torch.manual_seed(42)

        inferer_path = "inferers." + 'inferer_' + config['model_name']
        inferer = importlib.import_module(inferer_path)
        inferer.run(config, data=config['data'])


