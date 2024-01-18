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
import pickle

from train_1805002 import Layer, DenseLayer, ReLULayer, SoftmaxLayer, dropout
from train_1805002 import one_hot_encode, FNNModel


if __name__ == "__main__":
    np.random.seed(1)
    epsilon = 1e-15

    train_validation_dataset = ds.EMNIST(root='./data', split='letters',
                                train=True,
                                transform=transforms.ToTensor(),
                                download=False)


    # independent_test_dataset = ds.EMNIST(root='./data',
    #                     split='letters',
    #                             train=False,
    #                             transform=transforms.ToTensor())
    
    with open("ids7.pickle", 'rb') as ids7:
        independent_test_dataset = pickle.load(ids7)

    test_loader = torch.utils.data.DataLoader(independent_test_dataset, batch_size=1024, shuffle=True)

    with open("model_1805002.pkl", "rb") as f:
        fnn_model = pickle.load(f)
        
        
    # test on independent test set
    test_y_true_list = []
    test_y_pred_list = []

    for X, y in test_loader:
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

        #compute accuracy
        predictions = np.argmax(AL, axis=0)
        labels = np.argmax(output, axis=0)

        #populate the lists for confusion matrix
        test_y_true_list.extend(labels)
        test_y_pred_list.extend(predictions)
        
    test_y_true_list = np.array(test_y_true_list)
    test_y_pred_list = np.array(test_y_pred_list)
    

    test_accuracy = accuracy_score(test_y_true_list, test_y_pred_list)
    test_f1_score = f1_score(test_y_true_list, test_y_pred_list, average='macro')

    print("Test Accuracy: {:.4f}".format(test_accuracy))
    print("Test F1 Score: {:.4f}".format(test_f1_score))

    #confusion matrix
    cm = confusion_matrix(test_y_true_list, test_y_pred_list)
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig(f"plots/test_confusion_matrix.png")

