import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder 
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


dataset1 = pd.read_csv('datasets/1/WA_Fn-UseC_-Telco-Customer-Churn.csv')
dataset2 = pd.read_csv('datasets/2/adult.data')
dataset3 = pd.read_csv('datasets/3/creditcard.csv')

dataset = dataset1
#dataset = dataset2
#dataset = dataset3


## check missing data and data type of columns
# print(dataset.info())
# print(dataset.isnull().sum())
## unique value count of each column
# for column in dataset.columns:
#     print(f"{column}: {dataset[column].nunique()}, datatype: {dataset[column].dtype}")


# extract X dataframe keeping column labels intact
X = dataset.iloc[:, :-1]
# extract y dataframe keeping column labels intact
Y = dataset.iloc[:, -1]

#label encode Y
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

X.head()


# return list of those column names
def get_binary_columns(X_binary):
    binary_columns = []
    for column in X_binary.columns:
        if X_binary[column].dtype == 'object' and X_binary[column].nunique() == 2:
            binary_columns.append(column)
    return binary_columns


def get_test_train_split(X, Y):
    # test train split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, 
                                                        test_size = 0.2, random_state = 0)
    return X_train, X_test, Y_train, Y_test

def global_labelEncode_oneHotEncode(X_df, columns_to_hot_encode, columns_to_label_encode):
    #one hot encode
    one_hot_encoder = OneHotEncoder(sparse_output=False, drop='first')
    X_df_encoded = one_hot_encoder.fit_transform(X_df[columns_to_hot_encode])
    X_df = X_df.drop(columns_to_hot_encode, axis=1)
    
    print(f"shape of X_df before concat: {X_df.shape}")
    
    X_df = pd.concat([X_df, pd.DataFrame(X_df_encoded)], axis=1)
    
    print(f"shape of X_df after concat: {X_df.shape}")
    
    #label encode
    label_encoder = LabelEncoder()
    for column in columns_to_label_encode:
        X_df[column] = label_encoder.fit_transform(X_df[column])
        
    return X_df


def global_preprocess(X,Y, 
                      columns_to_hot_encode=None, 
                      columns_to_label_encode=None,
                      columns_to_normalize=None,
                      fill_missing_type='mean'):
    
    # one hot encode and label encode before splitting
    X = global_labelEncode_oneHotEncode(X, columns_to_hot_encode, columns_to_label_encode)
    X_df_train, X_df_test, Y_train, Y_test = get_test_train_split(X, Y)
    
    
    # #convert to numeric and fill missing values
    X_df_train = X_df_train.apply(pd.to_numeric, errors='coerce')
    X_df_test = X_df_test.apply(pd.to_numeric, errors='coerce')
    if fill_missing_type == 'zero':
        X_df_train = X_df_train.fillna(0)
        X_df_test = X_df_test.fillna(0)
    else:    
        X_df_train = X_df_train.fillna(X_df_train.mean())
        X_df_test = X_df_test.fillna(X_df_test.mean())
            
    # #normalize
    scaler = MinMaxScaler()
    X_df_train[columns_to_normalize] = scaler.fit_transform(X_df_train[columns_to_normalize])
    X_df_test[columns_to_normalize] = scaler.transform(X_df_test[columns_to_normalize])
    
    return X_df_train, X_df_test, Y_train, Y_test
    

def preprocess_dataset1(X, Y):
    X_df = X.drop('customerID', axis=1)
    
    X_df['MultipleLines'] = X_df['MultipleLines'].replace('No phone service', 'No')
    internet_columns = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    
    for i in internet_columns:
        X_df[i] = X_df[i].replace('No internet service', 'No')
        
    columns_to_hot_encode = ['InternetService', 'Contract', 'PaymentMethod']
    # columns_to_label_encode = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'OnlineSecurity', 
    #                            'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
    #                            'PaperlessBilling']
    cols = get_binary_columns(X_df)    
    columns_to_normalize = ['tenure', 'MonthlyCharges', 'TotalCharges']
    
    
    
    X_df_train, X_df_test, Y_train, Y_test = global_preprocess(X_df, Y, 
                                                            columns_to_hot_encode=columns_to_hot_encode,
                                                                columns_to_label_encode=cols,
                                                                columns_to_normalize=columns_to_normalize,
                                                                fill_missing_type='mean')

    return X_df_train, X_df_test, Y_train, Y_test


## Split and Preprocess dataset
X_train, X_test, Y_train, Y_test = preprocess_dataset1(X, Y)

def entropy(labels):
    unique_labels, counts = np.unique(labels, return_counts=True)
    probabilities = counts / len(labels)
    entropy_value = -np.sum(probabilities * np.log2(probabilities))
    return entropy_value

def feature_selection_by_IG(X_train, X_test, Y_train, num_features=-1):
    base_entropy = entropy(Y_train)
    print(f"base entropy: {base_entropy}")
    
    preprocessed_X = X_train.copy()
    
    # calculate entropy for each feature
    entropies = []
    InfoGain = []
    for column in preprocessed_X.columns:
        feat_value, feat_value_counts = np.unique(preprocessed_X[column], return_counts=True)
        weighted_feature_entropy = 0
        
        for value, count in zip(feat_value, feat_value_counts):
            #weighted_feature_entropy += count * entropy(Y_train[preprocessed_X[column] == value])
            weighted_feature_entropy += (count / len(preprocessed_X)) * entropy(Y_train[preprocessed_X[column] == value])
            
            
        entropies.append(weighted_feature_entropy)
        InfoGain.append(base_entropy - weighted_feature_entropy)
        #print(f"entropy for column '{column}': {weighted_feature_entropy} and information gain: {base_entropy - weighted_feature_entropy}" )
        
    # sort by InfoGain
    sorted_indices = np.argsort(InfoGain)[::-1]
    sorted_IG = np.sort(InfoGain)[::-1]
    sorted_columns = preprocessed_X.columns[sorted_indices]
    
    #print(sorted_IG)
    
    if num_features == -1:
        num_features = len(sorted_columns)
        
    # return dataframe with selected features and extract the same features from the test set
    truncated_X_train = pd.DataFrame(preprocessed_X[sorted_columns[:num_features]], columns=sorted_columns[:num_features])
    
    # only select those features from test set which are present in the truncated_X_train
    truncated_X_test = pd.DataFrame(X_test[sorted_columns[:num_features]], columns=sorted_columns[:num_features])
    
    return truncated_X_train, truncated_X_test


## Final feature selection
trunc_X_train, trunc_X_test = feature_selection_by_IG(X_train, X_test, Y_train, num_features=10)

print(f"shape of X_train: {trunc_X_train.shape}")
print(f"shape of X_test: {trunc_X_test.shape}")        
    

class modified_regressor:
    def __init__(self, X_train, X_test, Y_train, Y_test, data_point_weights, threshold, max_feature_count=-1, learning_rate=0.1):
        self.X = X_train
        self.Y = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        
        self.y_hat = None
        self.data_point_weights = data_point_weights
        self.weights = None
        self.max_feature_count = max_feature_count
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.selected_features = None
        
        self.sorted_indices = None
        self.sorted_IG = None
        self.sorted_columns = None
            
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def binary_cross_entropy(self, y, y_hat):
        return -(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
    
    def gradient_descent(self):
        z = np.dot(self.X, self.weights)
        y_hat = self.sigmoid(z)
        error = y_hat - self.Y
        gradient = np.dot(self.X.T, error) / len(self.Y)
        self.weights -= self.learning_rate * gradient
    
    def train(self):        
        self.weights = np.zeros(self.X.shape[1])
        
        if self.threshold is not None and self.threshold > 0:
            error = float('inf')
            iteration = 0
            
            while error > self.threshold:
                self.gradient_descent()
                iteration += 1
                z = np.dot(self.X, self.weights)
                y_hat = self.sigmoid(z)
                sum_error = np.sum(self.binary_cross_entropy(self.Y, y_hat))
                error = sum_error / len(self.Y)
        else:
            steps = 5000
            for i in range(steps):
                self.gradient_descent(self.X, self.Y)
                z = np.dot(self.X, self.weights)
                y_hat = self.sigmoid(z)
                sum_error = np.sum(self.binary_cross_entropy(self.Y, y_hat))
                error = sum_error / len(self.Y)
        
        print(f"total iterations: {iteration}")        
        #return self.weights, error, iteration
        
    def predict(self):
        z = np.dot(self.X_test, self.weights)
        y_hat = np.round(self.sigmoid(z))
        self.y_hat = y_hat
        return y_hat
    
    def accuracy(self):
        return (np.sum(self.y_hat == self.Y_test) / len(self.Y_test))*100
                   


data_point_weights = np.ones(len(trunc_X_train)) / len(trunc_X_train)
regressor = modified_regressor(trunc_X_train, trunc_X_test, Y_train, Y_test, data_point_weights, 0.5, -1)
regressor.train()
regressor.predict()
#calculate accuracy
accuracy = regressor.accuracy()
print(f"accuracy: {accuracy}")



def adaboost(examples_X, examples_Y, L_weak, K, num_features=-1):
    """
    Parameters
    ----------
    
    examples : set of N examples
    L_weak : weak learner (logistic regression)
    K : number of weak learners to use
    
    ------------------
    """
    
    data_point_weights = np.ones(len(examples_X)) / len(examples_X)
    hypothesises = []
    hypo_weights = []
    
    #feature selection
    examples_X, examples_Y = feature_selection_by_IG(examples_X, examples_Y, num_features)
    
    for i in range(K):
        #resampling data
        indices = np.random.choice(len(examples_X), len(examples_X), p=data_point_weights)
        resampled_examples_X = examples_X[indices]
        resampled_examples_Y = examples_Y[indices]
        
        # initialize regressor
        regressor = modified_regressor(resampled_examples_X, resampled_examples_Y, data_point_weights, 0.5, -1)
        #hypothesises.append(regressor)
        
        error = 0
        for j in range(len(examples_X)):
            if regressor.predict(examples_X[j]) != examples_Y[j]:
                error += data_point_weights[j]
            
        if error > 0.5:
            continue
        else:
            hypothesises.append(regressor)
        
        for j in range(len(examples_X)):
            if regressor.predict(examples_X[j]) == examples_Y[j]:
                data_point_weights[j] *= error / (1 - error)
        
        # normalizing weights
        data_point_weights /= np.sum(data_point_weights)
        
        hypo_weights.append(np.log2((1 - error) / error))
        
    return hypothesises, hypo_weights

def weighted_majority(hypothesises, hypo_weights, examples_X):
    predictions = []
    
    #normalize hypothesis weights
    hypo_weights /= np.sum(hypo_weights)
    
    for i in range(len(examples_X)):
        prediction = 0
        for j in range(len(hypothesises)):
            prediction += hypo_weights[j] * hypothesises[j].predict(examples_X[i])  
        prediction /= len(hypothesises)
        predictions.append(np.round(prediction))
    return predictions

    
    
    