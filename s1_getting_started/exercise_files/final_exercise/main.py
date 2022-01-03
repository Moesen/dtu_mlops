import argparse
import sys
from os import path

import torch
from torch import nn
from torch import optim

from data import mnist
from model import MyAwesomeModel


class TrainOREvaluate(object):
    """ Helper class that will help launch class methods as commands
        from a single script
    """
    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Script for either training or evaluating",
            usage="python main.py <command>"
        )
        parser.add_argument("command", help="Subcommand to run")
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print('Unrecognized command')
            
            parser.print_help()
            exit(1)
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()
    
    def train(self):
        print("Training day and night")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--lr', default=0.1, type=float, help="Learning Rate for optimizer")
        parser.add_argument('--ep', default=10, type=int, help="Number of epochs")
        parser.add_argument('--fn', default="temp", type=str, help="Filename")
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        
        # TODO: Implement training loop here
        model = MyAwesomeModel()
        train_x, train_Y = mnist(train=True)

        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        epochs = args.ep
        train_losses, test_losses, test_accuracy = [], [], []
        for e in range(epochs):
            running_loss = 0
            for images, labels in zip(train_x, train_Y):
                optimizer.zero_grad()
                images = images.resize_(images.size()[0], 784)
                print(images.shape)
                sys.exit()
                output = model.forward()

                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()

        checkpoint = {
            "input_size": 784,
            "output_size": 10,
            "hidden_layers": [each.out_features for each in model.hidden_layers],
            "state_dict": model.state_dict()
        }

        filename = args.fn
        torch.save(checkpoint, path.join("model", filename + ".pth"))


    def evaluate(self):
        print("Evaluating until hitting the ceiling")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('load_model_from', default="")
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        
        # TODO: Implement evaluation logic here
        model = torch.load(args.load_model_from)
        test_x, test_Y = mnist(train=False)

    

if __name__ == '__main__':
    TrainOREvaluate()
    
    
    
    
    
    
    
    
    