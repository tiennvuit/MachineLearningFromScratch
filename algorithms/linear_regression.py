import os
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from utils.read_data import read_data
from base_algorithm import AlgorithmML
from utils.utils import print_info, load_config, plot_progress
from utils.normalization import min_max_normalization


class LinearRegression(AlgorithmML):
    def __init__(self, args=None) -> None:
        super().__init__()
        args = load_config(path=args['config'])
        self.batch_size = args['batch_size']
        self.epochs = args['epochs']
        self.learning_rate = args['learning_rate']
        self.optimizer = args['optimizer']
        self.save_freq = args['save_freq']

        # Call some functions
        self.build_model()


    def __str__(self) -> str:
        return super().__str__()


    def build_model(self):
        # Build the function h(x) = W*x, W = [w_{n}, w_{n-1}, w_{n-2}, ..., w_{1}, w_{0}]
        np.random.seed(seed=18521489)
        self.parameters['W'] = np.random.rand(9)
        # self.parameters['b'] = np.random.rand(1)


    def learn(self, X:np.array, y:np.array, output_log='logs/linear_regression.csv'):
        
        # Define number of interation for each epoch
        self.iters = len(y) // self.batch_size

        # Performce learning progress
        for epoch in range(self.epochs):
            print("Current epoch: {}".format(epoch))
    
            # Loop through each iterations
            for iter in tqdm(range(self.iters)):
                
                # Get the random indices
                idx_current_batch = np.random.choice(range(len(y)), self.batch_size)
                mini_batch = np.hstack((X[idx_current_batch],np.zeros((X[idx_current_batch].shape[0],1))))
                y_true = y[idx_current_batch]
    
                # Feed fowarding
                y_pred = np.sum(self.parameters['W']*mini_batch, axis=1)

                # Computer gradient
                gradient = np.sum(np.expand_dims((y_pred-y_true),1)*mini_batch, axis=0)/len(idx_current_batch)
                
                # Update paramters
                self.parameters['W'] = self.parameters['W'] - self.learning_rate*gradient
            
            if epoch % self.save_freq == 0 and epoch != 0 :
                # Save the model to disk
                print("Epoch: {}, parameters: {}".format(epoch, self.parameters['W']))


def main(args):
    
    # Read data from local disk
    print("[info] Reading data from disk ...")
    X, y = read_data(path=args['data_path'])
    
    # Normalize data
    # X = min_max_normalization(X=X)
    
    # Build model
    print("[info] Building model ...")
    model = LinearRegression(args=args)

    # Fitting model with data
    print("[info] learning ...")
    model.learn(X=X, y=y, output_log='logs/linear_regression.csv')

    # Plot the training progress
    plot_progress(path='log', save=True)

    # Evaluate on test set
    model.evaluate()

    # End the process
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simple linear regression algorithm.')
    parser.add_argument('--data_path', type=str, default='../../data/housing.csv',
                        help='The directory path of data')
    parser.add_argument('--config', type=str, default='config/linear_regression.cfg',
                        help='The config path container all model configuration.')
    args = vars(parser.parse_args())
    
    print_info(args=args)

    main(args)