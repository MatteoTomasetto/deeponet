import numpy as np
import torch
import deepxde as dde
#from scipy import io
#from sklearn.preprocessing import StandardScaler

class DeepONet:

    def __init__(self, pair_id, config, train_data, init_data = None, prediction_horizon_steps = None):

        self.pair_id = pair_id

        self.train_data = train_data
        self.n = train_data[0].shape[0]
        self.m = train_data[0].shape[1]
        self.x = np.arange(0, self.n).astype(np.float32) ## TODO: Fix with domain info
        self.t = np.arange(0, self.m) ## TODO: Fix with time-step info

        self.init_data = init_data
        self.M = init_data.shape[1] if init_data is not None else 0

        self.prediction_horizon_steps = prediction_horizon_steps if prediction_horizon_steps is not None else self.m      

        self.branch_layers = config['model']['branch_layers']
        self.trunk_layers = config['model']['trunk_layers']
        self.branch_neurons = config['model']['branch_neurons']
        self.trunk_neurons = config['model']['trunk_neurons']
        self.activation = config['model']['activation']
        self.initialization = config['model']['initialization']
        self.optimizer = config['model']['optimizer']
        self.learning_rate = config['model']['learning_rate']
        self.epochs = config['model']['epochs']
        self.batch_size = config['model']['batch_size']

        self.branch = [self.n] + [self.branch_neurons] * self.branch_layers
        self.trunk = [1] + [self.trunk_neurons] * self.trunk_layers


    def get_data(self):

        data = dde.data.TripleCartesianProd(X_train = (self.train_data[0][:,:-1].T.astype(np.float32), self.x.reshape(-1, 1)), y_train = self.train_data[0][:,1:].T.astype(np.float32), X_test = (self.train_data[0][:,:-1].T.astype(np.float32), self.x.reshape(-1, 1)), y_test = self.train_data[0][:,1:].T.astype(np.float32))
        return data
        
        # OK for pair_id = 1 and KS
        # data_test is equal to data_train since it is not available. Add valid_test?
        # TODO: Add time-step info
        # TODO: Add past values (lag)


    def get_model(self):
        # OK for pair_id = 1 and KS

        deeponet = dde.nn.DeepONetCartesianProd(self.branch, self.trunk, self.activation, self.initialization)
        return deeponet


    def train(self):
        # OK for pair_id = 1 and KS

        data = self.get_data()
        deeponet = self.get_model()
        
        model = dde.Model(data, deeponet)
        model.compile(self.optimizer, lr = self.learning_rate)
        _, _ = model.train(epochs = self.epochs, batch_size = self.batch_size)

        return model


    def predict(self):
        # OK for pair_id = 1
        # What if we get NaN in the predictions?

        model = self.train()
        predictions = np.zeros((self.n, self.prediction_horizon_steps + 1))
        predictions[:,0] = self.train_data[0][:,-1]
        for i in range(self.prediction_horizon_steps):           
            predictions[:,i+1] = model.predict((predictions[:,i].reshape(1,-1), self.x.reshape(-1, 1)))

        return predictions[:,1:]

# TODO: CHECK EXTENSIONS IN THE DEEPONET REPO LIKE THE FOLLOWING
#     net = dde.maps.DeepONetCartesianProd(
#         [m, 128, 128, 128, 128], [1, 128, 128, 128], "tanh", "Glorot normal"
#     )
#     net.apply_feature_transform(periodic)
#     scaler = StandardScaler().fit(y_train)
#     std = np.sqrt(scaler.var_.astype(np.float32))
#     def output_transform(inputs, outputs):
#         return outputs * std + scaler.mean_.astype(np.float32)
#     net.apply_output_transform(output_transform)
#decay = ("inverse time", epochs // 5, 0.5)
#model.compile("adam", lr=lr, metrics=["mean l2 relative error"], decay=decay)



