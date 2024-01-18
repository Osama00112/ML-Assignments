import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import torchvision.datasets as ds
import torchvision.transforms as transforms
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score


np.random.seed(1)
epsilon = 1e-15

train_validation_dataset = ds.EMNIST(root='./data', split='letters',
                              train=True,
                              transform=transforms.ToTensor(),
                              download=False)


independent_test_dataset = ds.EMNIST(root='./data',
                       split='letters',
                             train=False,
                             transform=transforms.ToTensor())

class Layer:
    def __init__(self):
        self.input = None
        self.output = None
        
    def forward(self, input):
        raise NotImplementedError
    
    def backward(self, output):
        raise NotImplementedError
    
    def gradient_descent(self, learning_rate):
        raise NotImplementedError
    
    def get_cache(self):
        raise NotImplementedError
    
    
def optimize_adam(param, grads, m, v, t, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
    t += 1
    
    m = beta1 * m + (1 - beta1) * grads
    v = beta2 * v + (1 - beta2) * np.square(grads)
    
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    
    param -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
    
    return param, m, v, t


class DenseLayer(Layer):
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        
        # self.W = np.random.randn(output_size, input_size) * 0.01
        # self.b = np.zeros((output_size, 1))
        
        #xavier initialization
        variance = 2 / (input_size + output_size)
        self.W = np.random.randn(output_size, input_size) * np.sqrt(variance)
        self.b = np.zeros((output_size, 1))
        
        self.A_prev = None
        self.Z = None
        
        self.dW = None
        self.db = None
        
        self.A = None
        
        #adam parameters
        self.mw = np.zeros_like(self.W)
        self.vw = np.zeros_like(self.W)
        self.mb = np.zeros_like(self.b)
        self.vb = np.zeros_like(self.b)
        
        self.tw = 1
        self.tb = 1
        
        
        
    def forward(self, A_prev):
        self.A_prev = A_prev   
        self.Z = np.dot(self.W, self.A_prev) + self.b
        self.A = self.Z
        

    def backward(self, dZ):
        #if dZ dim is 1, then reshape it
        if len(dZ.shape) == 1:
            dZ = dZ.reshape(-1, 1)
            
        m = self.A_prev.shape[1]
        
        self.dW = np.dot(dZ, self.A_prev.T) / m
        self.db = np.sum(dZ, axis=1, keepdims=True) / m
        
        self.dA_prev = np.dot(self.W.T, dZ)
        
        return self.dA_prev
        
    def get_cache(self):
        return [self.W, self.b]
    
    def gradient_descent(self, learning_rate):
        self.W, self.mw, self.vw, self.tw = optimize_adam(self.W, self.dW, self.mw, self.vw, self.tw, learning_rate)
        self.b, self.mb, self.vb, self.tb = optimize_adam(self.b, self.db, self.mb, self.vb, self.tb, learning_rate)
        
        
        
class ReLULayer(Layer):
    def forward(self, Z):
        self.Z = Z
        self.A = np.maximum(0, Z)
        
    def backward(self, dA):
        dZ = dA * (self.Z > 0)
        return dZ
    
    def gradient_descent(self, learning_rate):
        pass
    
    def get_cache(self):
        return []
    
class SigmoidLayer(Layer):
    def forward(self, Z):
        self.Z = Z
        self.A = 1 / (1 + np.exp(-Z))
        
    def backward(self, dA):
        g_z = self.A
        dZ = dA * g_z * (1 - g_z)
        
        return dZ
    
    def gradient_descent(self, learning_rate):
        pass
    
    def get_cache(self):
        return []
        
        
class SoftmaxLayer(Layer):
    def forward(self, Z):
        exps = np.exp(Z - np.max(Z))
        self.A = exps / np.sum(exps, axis=0)
        
    def backward(self, dA):
        return dA
    
    
    def gradient_descent(self, learning_rate):
        pass
    
    def get_cache(self):
        return []
    
    
class dropout(Layer):
    def __init__(self, drop_rate):
        self.drop_rate = drop_rate
        self.mask = None
        
    def forward(self, A):
        random_matrix = np.random.rand(*A.shape)
        self.mask = random_matrix < self.drop_rate
        self.A = np.multiply(A, self.mask)
        
    def backward(self, dA):
        dA = np.multiply(dA, self.mask)
        dA /= self.drop_rate
        
        return dA
    
    def gradient_descent(self, learning_rate):
        pass
    
    def get_cache(self):
        return []   
    
    
class FNNModel:
    def __init__(self, network_layers):
        self.layers = []
        for i, layer in enumerate(network_layers):
            if isinstance(layer, DenseLayer):
                self.layers.append(DenseLayer(layer.input_size, layer.output_size))
            elif isinstance(layer, ReLULayer):
                self.layers.append(ReLULayer())
            elif isinstance(layer, SigmoidLayer):
                self.layers.append(SigmoidLayer())
            elif isinstance(layer, SoftmaxLayer):
                self.layers.append(SoftmaxLayer())
            elif isinstance(layer, dropout):
                self.layers.append(dropout(layer.drop_rate))
            else:
                raise ValueError("Invalid layer type")
        
        self.parameters = []
        for layer in self.layers:
            self.parameters.extend(layer.get_cache())

    def forward_propagation(self, X):
        A = X
        for layer in self.layers:
            layer.forward(A)
            A = layer.A
    
    def compute_cost(self, AL, Y, loss_type='binary'):
        m = Y.shape[1]
        
        if loss_type == 'binary':
            cost = - (1/m) * np.sum(Y * np.log(AL + epsilon) + (1 - Y) * np.log(1 - AL + epsilon))
        elif loss_type == 'multiclass':
            AL = np.clip(AL, epsilon, 1 - epsilon)
            cost = np.sum(-np.log(AL) * Y) / m
        else:
            raise ValueError("Invalid loss_type")
            
        cost = np.squeeze(cost)
        return cost
    
    def compute_last_layer_dAL(self, AL, Y, loss_type='binary'):
    
        if loss_type == 'binary':
            self.dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
        elif loss_type == 'multiclass':
            self.dAL = AL - Y
        else:
            raise ValueError("Invalid loss_type")

    def backward_propagation(self, X, Y, AL, loss_type='binary'):
        m = X.shape[1]
        self.compute_last_layer_dAL(AL, Y, loss_type=loss_type)
        
        dAL_last_layer = self.dAL
        
        for layer in reversed(self.layers):
            dAL = layer.backward(dAL_last_layer)
            dAL_last_layer = dAL
        
    def update_parameters(self, learning_rate, optimizer="gradient_descent"):
        for layer in self.layers:
            layer.gradient_descent(learning_rate)
    

    def train(self, X, Y, epochs=3000, learning_rate=0.01, 
              loss_type = "binary", optimizer="gradient_descent",
              print_cost=False):
        costs = []
        for i in range(epochs):
            self.forward_propagation(X)
            
            if not isinstance(self.layers[-1], (ReLULayer, SoftmaxLayer)):
                raise ValueError("Last layer should be an activation layer")

            
            AL = self.layers[-1].A
            cost = self.compute_cost(AL, Y, loss_type=loss_type)
            costs.append(cost)
            
            self.backward_propagation(X, Y, AL, loss_type=loss_type)
            self.update_parameters(learning_rate, optimizer=optimizer)
            
            if print_cost and i % 100 == 0:
                print("Cost after iteration {}: {}".format(i, cost))
                
        return costs
    
    def predict(self, X):
        self.forward_propagation(X)
        AL = self.layers[-1].A
        
        return AL
    
    def evaluate(self, X, Y):
        AL = self.predict(X)
        predictions = np.argmax(AL, axis=0)
        labels = np.argmax(Y, axis=0)
        
        accuracy = np.mean(predictions == labels)
        
        return accuracy   
    
def one_hot_encode(y, num_classes):
    y_labels = y-1
    y_one_hot = np.zeros((num_classes, y.shape[0]))
    y_one_hot[y_labels, np.arange(y.shape[0])] = 1
    
    return y_one_hot


if __name__ == "__main__":
    
    # splitting the data into train and validation sets
    train_size = int(0.85 * len(train_validation_dataset))
    validation_size = len(train_validation_dataset) - train_size
    train_dataset, validation_dataset = torch.utils.data.random_split(train_validation_dataset, [train_size, validation_size])

    # dataloader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1024, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=1024, shuffle=True)

    test_loader = torch.utils.data.DataLoader(independent_test_dataset, batch_size=1024, shuffle=True)



    #train again with best parameters
    network = [
        DenseLayer(784, 1024),
        ReLULayer(),
        dropout(0.15),
        DenseLayer(1024, 26),
        SoftmaxLayer()
    ]

    # Create the FNN model
    fnn_model = FNNModel(network)

    learning_rate = 0.005
    epochs = 20
    print_cost = True
    loss_type = 'multiclass'
    optimizer = "adam"

    #update train loader to be the entire dataset
    train_loader = torch.utils.data.DataLoader(train_validation_dataset, batch_size=1024, shuffle=True)

    for epoch in tqdm(range(epochs)):
        for X, y in train_loader:
            X = X.view(X.shape[0], -1).numpy().T
            y = one_hot_encode(y.numpy(), 26)

            # Train the FNN model
            input = X
            output = y

            for layer in fnn_model.layers:
                layer.forward(input)
                input = layer.A

            #compute loss
            AL = fnn_model.layers[-1].A
            cost = fnn_model.compute_cost(AL, output, loss_type=loss_type)
            #rain_loss += cost

            #backward propagation
            fnn_model.backward_propagation(X, y, AL, loss_type=loss_type)
            fnn_model.update_parameters(learning_rate, optimizer=optimizer)
            
    
        print("Epoch: {}".format(epoch+1))
        print("Training Loss: {:.4f}".format(cost))
        print("====================================")
            

    import pickle

    with open("model_1805002.pkl", "wb") as f:
        pickle.dump(fnn_model, f)
        
    
"""

hidden_layer = [256, 512, 1024]

for h in hidden_layer:
    network = [
        DenseLayer(784, h),
        ReLULayer(),
        dropout(0.15),
        DenseLayer(h, 26),
        SoftmaxLayer()
    ]

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1024, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=1024, shuffle=True)

    test_loader = torch.utils.data.DataLoader(independent_test_dataset, batch_size=1024, shuffle=True)

    learning_rates = [5e-3, 1e-3, 5e-4, 1e-4]

    best_validation_mark = 0
    best_learning_rate = 0
    best_parameters = None

    for learning_rate in learning_rates:
        print("Learning Rate: {}".format(learning_rate))
        print("====================================")
        # Create the FNN model
        fnn_model = FNNModel(network)
        
        train_loss_list = []
        validation_loss_list = []
        train_accuracy_list = []
        validation_accuracy_list = []
        validation_f1_list = []
        
        
        
        epochs = 10
        print_cost = True
        loss_type = 'multiclass'
        optimizer = "adam"
        
        for epoch in tqdm(range(epochs)):
            train_loss = 0
            train_accuracy = 0
            validation_loss = 0
            validation_accuracy = 0
            
            for X, y in train_loader:
                X = X.view(X.shape[0], -1).numpy().T
                y = one_hot_encode(y.numpy(), 26)
                
                # Train the FNN model
                input = X
                output = y
                
                for layer in fnn_model.layers:
                    layer.forward(input)
                    input = layer.A
          
                
                #compute loss
                AL = fnn_model.layers[-1].A
                cost = fnn_model.compute_cost(AL, output, loss_type=loss_type)
                train_loss += cost
                
                #compute accuracy
                predictions = np.argmax(AL, axis=0)
                labels = np.argmax(output, axis=0)
                accuracy = np.mean(predictions == labels)
                train_accuracy += accuracy
                
                #backward propagation
                fnn_model.backward_propagation(X, y, AL, loss_type=loss_type)
                fnn_model.update_parameters(learning_rate, optimizer=optimizer)
                
            train_loss /= len(train_loader)
            train_accuracy /= len(train_loader)
            
            train_loss_list.append(train_loss)
            train_accuracy_list.append(train_accuracy)
            
            
            validation_y_true_list = []
            validation_y_pred_list = []
            
            
            for X, y in validation_loader:
                X = X.view(X.shape[0], -1).numpy().T
                y = one_hot_encode(y.numpy(), 26)
                
                # Train the FNN model
                input = X
                output = y
                
                for layer in fnn_model.layers:
                    #if dropout layer, skip
                    if isinstance(layer, dropout):
                        continue
                    layer.forward(input)
                    input = layer.A
                
                #compute loss
                AL = fnn_model.layers[-1].A
                cost = fnn_model.compute_cost(AL, output, loss_type=loss_type)
                validation_loss += cost
                
                #compute accuracy
                predictions = np.argmax(AL, axis=0)
                labels = np.argmax(output, axis=0)
                accuracy = np.mean(predictions == labels)
                validation_accuracy += accuracy
                
                #populate the lists for confusion matrix
                validation_y_true_list.extend(labels)
                validation_y_pred_list.extend(predictions)
                

                #backward propagation
                fnn_model.compute_last_layer_dAL(AL, output, loss_type=loss_type)     
                dAL = fnn_model.dAL   
                for layer in reversed(fnn_model.layers):
                    #if dropout layer, skip
                    if isinstance(layer, dropout):
                        continue
                    dAL = layer.backward(dAL)            
                fnn_model.update_parameters(learning_rate, optimizer=optimizer)
                
            
            validation_y_true_list = np.array(validation_y_true_list)
            validation_y_pred_list = np.array(validation_y_pred_list)
            
            validation_loss /= len(validation_loader)
            validation_accuracy /= len(validation_loader)
            validation_f1_score = f1_score(labels, predictions, average='macro')
            
            validation_loss_list.append(validation_loss)
            validation_accuracy_list.append(validation_accuracy)
            validation_f1_list.append(validation_f1_score)
            
            print("Epoch: {}".format(epoch+1))
            print("Training Loss: {:.4f}".format(train_loss))
            print("Validation Loss: {:.4f}".format(validation_loss))
            print("Training Accuracy: {:.4f}".format(train_accuracy))
            print("Validation Accuracy: {:.4f}".format(validation_accuracy))
            print("Validation F1 Score: {:.4f}".format(validation_f1_score))
            print("====================================")
            
            if validation_accuracy > best_validation_mark:
                best_validation_mark = validation_accuracy
                best_learning_rate = learning_rate
                best_parameters = fnn_model.parameters
                
        
            
            
            
        #plot the loss and accuracy curves

        plt.figure(figsize=(10, 10))
        plt.plot(train_loss_list, label="Train Loss", color="blue")
        plt.plot(validation_loss_list, label="Validation Loss", color="red")
        plt.title("Loss vs Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.savefig(f"plots/{h}_loss_{learning_rate}.png")
        plt.legend()

        plt.figure(figsize=(10, 10))
        plt.plot(train_accuracy_list, label="Train Accuracy", color="blue")
        plt.plot(validation_accuracy_list, label="Validation Accuracy", color="red")
        plt.title("Accuracy vs Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.savefig(f"plots/{h}_accuracy_{learning_rate}.png")
        plt.legend()     

        plt.figure(figsize=(10, 10))
        plt.plot(validation_f1_list, label="Validation F1 Score", color="red")
        plt.title("F1 Score vs Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("F1 Score")
        plt.savefig(f"plots/{h}_f1_score_{learning_rate}.png")
        plt.legend()
        
        
        #confusion matrix
        cm = confusion_matrix(validation_y_true_list, validation_y_pred_list)
        plt.figure(figsize=(10, 10))
        sns.heatmap(cm, annot=True, fmt="d")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.savefig(f"plots/{h}_confusion_matrix_{learning_rate}.png")
        plt.show()
        
        

    print("Best Learning Rate: {}".format(best_learning_rate))
    print("Best Validation Mark: {}".format(best_validation_mark))
    print("Best Parameters: {}".format(best_parameters))
    
"""