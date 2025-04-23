import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
import deepxde as dde
dde.backend.set_default_backend("pytorch")

class DeepONet:
    """
    Deep Operator Network (DeepONet) model.

    Attributes:
        pair_id (int): Identifier for the data pair to consider.
        
        lag (int): Number of past timesteps to consider in input.
        branch_layers (int): Number of layers in the branch network.
        trunk_layers (int): Number of layers in the trunk network.
        branch_neurons (int): Number of neurons in each layer of the branch network.
        trunk_neurons (int): Number of neurons in each layer of the trunk network.
        activation (str): Activation function to use in the networks.
        initialization (str): Initialization method for the networks.
        optimizer (str): Optimizer to use for training.
        learning_rate (float): Learning rate for the optimizer.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        
        train_data (np.ndarray): Training data.
        n (int): Number of spatial points.
        m (int): Number of time points.
        x (np.ndarray): Spatial coordinates.
        delta_t (float): Timestep length

        init_data (np.ndarray): Burn-in data for prediction.
        M (int): Number of past timesteps to consider in prediction.

        prediction_horizon_steps (int): Number of timesteps to predict.

        branch_input_dimension (int): Input dimension for the branch network.
        branch (list): Branch network architecture.
        trunk (list): Trunk network architecture.
        
        scaler (StandardScaler): Scaler for output data.
    """

    def __init__(self, pair_id, config, train_data, init_data = None, prediction_timesteps = None, delta_t = None):
        """
        Initialize the DeepONet model with the provided configuration.

        Args:
            config (Dict): Configuration dictionary containing method and parameters.
            train_data (np.ndarray): Training data.
            init_data (np.ndarray): Burn-in data for prediction.
            prediction_timesteps (np.ndarray): Prediction timesteps for the model.
            delta_t (float): Timestep length
    
        Raises:
            ValueError: If lag parameter is invalid.
        """     

        self.pair_id = pair_id

        if self.pair_id == 2 or self.pair_id == 4:
            print("Reconstruction task: 'lag' parameter equal to 0 since burn-in data not needed")
            self.lag = 0
        else:
            self.lag = config['model']['lag']
            if self.lag < 1:
                raise ValueError(f"Forecasting task: select a positive 'lag' parameter")
        
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

        self.train_data = train_data
        self.n = train_data[0].shape[0]
        self.m = train_data[0].shape[1]
        self.x = np.arange(0, self.n).astype(np.float32) ## TODO: Fix with domain info
        self.delta_t = delta_t
        if self.lag > self.m:
            raise ValueError(f"Select a 'lag' parameter smaller than the number of training timesteps ({self.m}).")
        
        self.init_data = init_data if init_data is not None else train_data[0][:,-self.lag:]
        self.M = init_data.shape[1] if init_data is not None else self.lag
        if self.lag > self.M:
            raise ValueError(f"Select a 'lag' parameter smaller than the number of burn-in timesteps ({self.M}).")
        
        self.prediction_horizon_steps = len(prediction_timesteps) if prediction_timesteps is not None else self.m
        
        self.branch_input_dimension = max(self.lag, 1) * self.n
        self.branch = [self.branch_input_dimension] + [self.branch_neurons] * self.branch_layers
        self.trunk_input_dimension = 1 if delta_t is None else 2
        self.trunk = [self.trunk_input_dimension] + [self.trunk_neurons] * self.trunk_layers


    def get_data(self) -> dde.data.triple.TripleCartesianProd:
        """
        Generate the data object for training by extracting input and output data.
        The input data is constructed by taking the past `lag` timesteps for each spatial point,
        and the output data is the corresponding future timesteps.
         
        Returns:
            dde.data.triple.TripleCartesianProd: data object for training the model.
        """

        trunk_input_data = self.x.reshape(-1, 1) if self.delta_t is None else np.column_stack((self.x, np.full_like(self.x, self.delta_t)))

        branch_input_data = np.zeros((len(self.train_data), (self.m - self.lag), self.branch_input_dimension), dtype = np.float32)
        for i in range(len(self.train_data)):
            for j in range(branch_input_data[i].shape[0]):
                branch_input_data[i,j] = self.train_data[i][:,j:j+max(self.lag, 1)].T.ravel().astype(np.float32)
        branch_input_data = branch_input_data.reshape(-1, self.branch_input_dimension)

        output_data = np.zeros((len(self.train_data), (self.m - self.lag), self.n), dtype = np.float32)
        for i in range(len(self.train_data)):
            output_data[i] = self.train_data[i][:,self.lag:].T.astype(np.float32)
        output_data = output_data.reshape(-1, self.n)

        self.scaler = StandardScaler().fit(output_data)
        data = dde.data.TripleCartesianProd(X_train = (branch_input_data, trunk_input_data), y_train = output_data, X_test = (branch_input_data, trunk_input_data), y_test = output_data)
        return data
        

    def get_model(self) -> dde.nn.pytorch.deeponet.DeepONetCartesianProd:
        """
        Generate the model object for training.
        The branch network takes the past `lag` timesteps for each spatial point,
        and the trunk network takes the corresponding spatial coordinates.
        The outputs of the branch and trunk networks are multiplied to get the prediction.

        Returns:
            dde.nn.pytorch.deeponet.DeepONetCartesianProd: model object for training.        
        """

        deeponet = dde.nn.DeepONetCartesianProd(self.branch, self.trunk, self.activation, self.initialization)
        return deeponet


    def train(self) -> dde.model.Model:
        """
        DeepONet training.

        Returns:
            dde.model.Model: trained model.  
        """

        data = self.get_data()
        deeponet = self.get_model()
        
        output_transform = lambda inputs, outputs: outputs * torch.from_numpy(np.sqrt(self.scaler.var_.astype(np.float32)) + self.scaler.mean_.astype(np.float32))
        deeponet.apply_output_transform(output_transform)
        
        model = dde.Model(data, deeponet)
        model.compile(self.optimizer, lr = self.learning_rate, metrics = ["mean l2 relative error"])
        _, _ = model.train(iterations = self.epochs, batch_size = self.batch_size)

        return model


    def predict(self) -> np.ndarray:
        """
        DeepONet predictions.

        Returns:
            np.ndarray: array of predictions.  
        """

        model = self.train()
        predictions = np.zeros((self.n, self.prediction_horizon_steps), dtype = np.float32)      
        trunk_input_data = self.x.reshape(-1, 1) if self.delta_t is not None else np.column_stack((self.x, np.full_like(self.x, self.delta_t)))

        if self.lag > 0:
            init_data = self.init_data[:,-self.lag:].T.ravel().astype(np.float32)
            for i in range(self.prediction_horizon_steps):           
                predictions[:,i] = model.predict((init_data.reshape(1, -1), trunk_input_data))
                init_data = np.concatenate((init_data[self.n:], predictions[:,i]))
        else:
            predictions = model.predict((self.init_data.T.astype(np.float32), trunk_input_data)).T

        return predictions
