import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder 
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


dataset_type = 3

missing_columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
                 'marital-status', 'occupation', 'relationship', 'race', 'sex',
                 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']

dataset1 = pd.read_csv('datasets/1/WA_Fn-UseC_-Telco-Customer-Churn.csv')

dataset2_train = pd.read_csv('datasets/2/adult.data', header=None, names=missing_columns, na_values=' ?')
dataset2_test = pd.read_csv('datasets/2/adult.test', header=None, names=missing_columns, na_values=' ?')
dataset2_train['income'] = dataset2_train['income'].str.replace('.', '')
dataset2_test['income'] = dataset2_test['income'].str.replace('.', '')
dataset2 = pd.concat([dataset2_train, dataset2_test])

dataset3 = pd.read_csv('datasets/3/creditcard.csv')

np.random.seed(1)




if dataset_type == 1:
    dataset = dataset1
elif dataset_type == 2:
    dataset = dataset2
elif dataset_type == 3:
    dataset = dataset3


## check missing data and data type of columns
# print(dataset.info())
# print(dataset.isnull().sum())
## unique value count of each column
# for column in dataset.columns:
#     print(f"{column}: {dataset[column].nunique()}, datatype: {dataset[column].dtype}")


# extract X dataframe keeping column labels intact
if dataset_type != 2:    
    # extract X dataframe keeping column labels intact
    X = dataset.iloc[:, :-1]
    # extract y dataframe keeping column labels intact
    Y = dataset.iloc[:, -1]

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
    one_hot_encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
    
    if columns_to_hot_encode is not None:
        X_df_encoded = one_hot_encoder.fit_transform(X_df[columns_to_hot_encode])
        X_df = X_df.drop(columns_to_hot_encode, axis=1)       
        X_df = pd.concat([X_df, pd.DataFrame(X_df_encoded)], axis=1)
    
    #label encode
    label_encoder = LabelEncoder()
    if columns_to_label_encode is not None:
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
    
    #label encode Y
    labelencoder_Y = LabelEncoder()
    Y = labelencoder_Y.fit_transform(Y)
    
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


def preprocess_dataset2(dataset_train, dataset_test):

    X_train = dataset_train.iloc[:, :-1]
    Y_train = dataset_train.iloc[:, -1]
    X_test = dataset_test.iloc[:, :-1]
    Y_test = dataset_test.iloc[:, -1]
    
    labelencoder_Y = LabelEncoder()
    Y_train = labelencoder_Y.fit_transform(Y_train)
    Y_test = dataset_test.iloc[:, -1]
    Y_test = labelencoder_Y.fit_transform(Y_test)
    
    # replace ? values with nan
    targert_columns = ['workclass', 'occupation', 'native-country']   
    X_train[targert_columns] = X_train[targert_columns].replace(' ?', np.nan)
    X_test[targert_columns] = X_test[targert_columns].replace(' ?', np.nan)
    
    
    # list columns that are of type object
    object_columns = X_train.select_dtypes(include=['object']).columns
    columns_to_hot_encode = object_columns
    X_train = global_labelEncode_oneHotEncode(X_train, columns_to_hot_encode, None)
    X_test = global_labelEncode_oneHotEncode(X_test, columns_to_hot_encode, None)
    
    X_train.columns = X_train.columns.astype(str)
    X_test.columns = X_test.columns.astype(str)
    # list columns that are of type numeric for nomralization
    numeric_columns = X_train.select_dtypes(include=['int64', 'float64']).columns
    columns_to_normalize = numeric_columns
    
    # #convert to numeric and fill missing values
    X_train = X_train.apply(pd.to_numeric, errors='coerce')
    X_test = X_test.apply(pd.to_numeric, errors='coerce')

    X_train = X_train.fillna(X_train.mean())
    X_test = X_test.fillna(X_test.mean())
            
    # #normalize
    scaler = MinMaxScaler()
    X_train[columns_to_normalize] = scaler.fit_transform(X_train[columns_to_normalize])
    X_test[columns_to_normalize] = scaler.transform(X_test[columns_to_normalize])
    
    return X_train, X_test, Y_train, Y_test

def preprocess_dataset3(X, Y):
    
    # list columns that are of type numeric for nomralization
    numeric_columns = X.select_dtypes(include=['int64', 'float64']).columns
    columns_to_normalize = numeric_columns
    
    #randomly select 2000 '0' class samples from Y and all '1' class samples
    concat_df = pd.concat([X, Y], axis=1)
    concat_df_0 = concat_df[concat_df['Class'] == 0]
    concat_df_1 = concat_df[concat_df['Class'] == 1]
    concat_df_0_sample = concat_df_0.sample(n=20000, random_state=0)
    concat_df_1_sample = concat_df_1
    concat_df_sample = pd.concat([concat_df_0_sample, concat_df_1_sample], axis=0)
    X = concat_df_sample.drop('Class', axis=1)
    Y = concat_df_sample['Class']
    
    X_train, X_test, Y_train, Y_test = global_preprocess(X, Y, 
                                                        columns_to_hot_encode=None,
                                                        columns_to_label_encode=None,
                                                        columns_to_normalize=columns_to_normalize,
                                                        fill_missing_type='mean')


    return X_train, X_test, Y_train, Y_test

if dataset_type == 1:
    X_train, X_test, Y_train, Y_test = preprocess_dataset1(X, Y)
elif dataset_type == 2:
    X_train, X_test, Y_train, Y_test = preprocess_dataset2(dataset2_train, dataset2_test)
elif dataset_type == 3:
    X_train, X_test, Y_train, Y_test = preprocess_dataset3(X, Y)

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

trunc_X_train, trunc_X_test = feature_selection_by_IG(X_train, X_test, Y_train, num_features=-1)

print(f"shape of X_train: {trunc_X_train.shape}")
print(f"shape of X_test: {trunc_X_test.shape}")
print(f"shape of Y_train: {Y_train.shape} and type: {type(Y_train)}")
print(f"shape of Y_test: {Y_test.shape} and type: {type(Y_test)}")  
    
class modified_regressor:
    def __init__(self, X_train, Y_train, 
                 data_point_weights, threshold=-1, max_feature_count=-1, learning_rate=0.1):
        self.X = X_train
        self.Y = Y_train
        self.X_test = None
        self.Y_test = None
        
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
        iteration = 0
        if self.threshold is not None and self.threshold > 0:
            error = float('inf')        
            
            steps = 1000
            for i in range(steps):
                self.gradient_descent()
                iteration += 1
                z = np.dot(self.X, self.weights)
                y_hat = self.sigmoid(z)
                sum_error = np.sum(self.binary_cross_entropy(self.Y, y_hat))
                error = sum_error / len(self.Y)
                if error <= self.threshold:
                    break
        else:
            steps = 1000
            for i in range(steps):
                iteration += 1
                self.gradient_descent()
                z = np.dot(self.X, self.weights)
                y_hat = self.sigmoid(z)
                sum_error = np.sum(self.binary_cross_entropy(self.Y, y_hat))
                error = sum_error / len(self.Y)
        
    def predict(self, X_test):
        self.X_test = X_test
        z = np.dot(self.X_test, self.weights)
        y_hat = np.round(self.sigmoid(z))
        self.y_hat = y_hat
        return y_hat
    
    def accuracy(self, Y_test):
        self.Y_test = Y_test
        return (np.sum(self.y_hat == self.Y_test) / len(self.Y_test))*100
                                   
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
        
    for i in range(K):
        #resampling data
        # resampling data
        indices = np.random.choice(len(examples_X), len(examples_X), p=data_point_weights)
        resampled_examples_X = examples_X.values[indices]
        resampled_examples_Y = examples_Y[indices]

        # initialize regressor
        regressor = L_weak(resampled_examples_X, resampled_examples_Y, data_point_weights, 0.5, num_features)
        regressor.train()
        y_hat = regressor.predict(examples_X)
        #hypothesises.append(regressor)
        
        error = 0
        for j in range(len(examples_X)):
            if y_hat[j] != examples_Y[j]:
                error += data_point_weights[j]
            
        if error > 0.5:
            continue
        else:
            hypothesises.append(regressor)
        
        for j in range(len(examples_X)):
            if y_hat[j] == examples_Y[j]:
                data_point_weights[j] *= error / (1 - error)
        
        # normalizing weights
        data_point_weights /= np.sum(data_point_weights)
        
        hypo_weights.append(np.log2((1 - error) / error))
        
    return hypothesises, hypo_weights

def weighted_majority(hypothesises, hypo_weights, examples_X):
    predictions = []
    #normalize hypothesis weights
    hypo_weights /= np.sum(hypo_weights)
    y_hats = np.zeros(len(examples_X))
    
    for i in range(len(hypothesises)):
        y_hats += hypo_weights[i] * hypothesises[i].predict(examples_X)
    
    predictions = np.round(y_hats)
    return predictions

#performance metrics
def perf_metrics(pred_Y, true_Y):
    TP = 0
    FN = 0
    FP = 0
    TN = 0
    
    for i in range(len(pred_Y)):
        if pred_Y[i] == true_Y[i] == 1:
            TP += 1
        elif pred_Y[i] == 1 and true_Y[i] == 0:
            FP += 1
        elif pred_Y[i] == 0 and true_Y[i] == 1:
            FN += 1
        else:
            TN += 1
            
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP + 1e-10)
    sensitivity = TP / (TP + FN + 1e-10)
    specificity = TN / (TN + FP + 1e-10)
    false_discovery_rate = FP / (FP + TP + 1e-10)
    F1_score = 2 / ((1 / (precision + 1e-10)) + (1 / (sensitivity + 1e-10)))
    
    return accuracy, precision, sensitivity, specificity, false_discovery_rate, F1_score
    
def print_stuffs(accuracy, sensitivity, specificity, precision, false_discovery_rate, F1_score):
    print(f"accuracy: {accuracy}")
    print(f"precision: {precision}")
    print(f"sensitivity: {sensitivity}")
    print(f"specificity: {specificity}")
    print(f"false_discovery_rate: {false_discovery_rate}")
    print(f"F1_score: {F1_score}")
    print("\n---------------\n")
    
# Without boosting
# if Y_train is series.series conver to ndarray
if type(Y_train) == pd.core.series.Series:
    Y_train = Y_train.values
if type(Y_test) == pd.core.series.Series:
    Y_test = Y_test.values

data_point_weights = np.ones(len(trunc_X_train)) / len(trunc_X_train)
regressor = modified_regressor(trunc_X_train, Y_train, data_point_weights)
regressor.train()

print(f"\nWithout Boosting:\n")
print(f"Training data")

y_pred = regressor.predict(trunc_X_train)
accuracy, precision, sensitivity, specificity, false_discovery_rate, F1_score = perf_metrics(y_pred, Y_train)
print_stuffs(accuracy, sensitivity, specificity, precision, false_discovery_rate, F1_score)

print(f"Test data")

y_pred = regressor.predict(trunc_X_test)
accuracy, precision, sensitivity, specificity, false_discovery_rate, F1_score = perf_metrics(y_pred, Y_test)
print_stuffs(accuracy, sensitivity, specificity, precision, false_discovery_rate, F1_score)




for K in range(5, 30, 5):
    print(f"K: {K}")
    hypothesises, hypo_weights = adaboost(trunc_X_train, Y_train,  
                                          modified_regressor, K)
    ## predictions
    print(f"Training data")
    y_hat = weighted_majority(hypothesises, hypo_weights, trunc_X_train)
    accuracy, precision, sensitivity, specificity, false_discovery_rate, F1_score = perf_metrics(y_hat, Y_train)
    print_stuffs(accuracy, sensitivity, specificity, precision, false_discovery_rate, F1_score)

    print(f"Test data")
    y_hat = weighted_majority(hypothesises, hypo_weights, trunc_X_test)
    accuracy, precision, sensitivity, specificity, false_discovery_rate, F1_score = perf_metrics(y_hat, Y_test)
    print_stuffs(accuracy, sensitivity, specificity, precision, false_discovery_rate, F1_score)





# #plot accuracy, F1 score of train and test set vs max_feature_count without boosting in the same plot 
# # for dataset1,2,3
# feature_counts = [5, 10, 15, len(trunc_X_train.columns)]
# feature_counts = [20, 40, 60, len(trunc_X_train.columns)]
# feature_counts = [5, 10, 20, len(trunc_X_train.columns)]

# accuracy_train = []
# precision_train = []
# sensitivity_train = []
# specificity_train = []
# false_discovery_rate_train = []
# F1_score_train = []

# accuracy_test = []
# precision_test = []
# sensitivity_test = []
# specificity_test = []
# false_discovery_rate_test = []
# F1_score_test = []

# for feature_count in feature_counts:
#     print(f"feature_count: {feature_count}")
#     trunc_X_train, trunc_X_test = feature_selection_by_IG(X_train, X_test, Y_train, num_features=feature_count)

#     data_point_weights = np.ones(len(trunc_X_train)) / len(trunc_X_train)
#     regressor = modified_regressor(trunc_X_train, Y_train, data_point_weights, max_feature_count=feature_count)
#     regressor.train()
    
#     print(f"\nWithout Boosting:\n")
#     print(f"Training data")
    
#     y_pred = regressor.predict(trunc_X_train)
#     accuracy, precision, sensitivity, specificity, false_discovery_rate, F1_score = perf_metrics(y_pred, Y_train)
#     accuracy_train.append(accuracy)
#     precision_train.append(precision)
#     sensitivity_train.append(sensitivity)
#     specificity_train.append(specificity)
#     false_discovery_rate_train.append(false_discovery_rate)
#     F1_score_train.append(F1_score)
#     print_stuffs(accuracy, sensitivity, specificity, precision, false_discovery_rate, F1_score)
    
#     print(f"Test data")
    
#     y_pred = regressor.predict(trunc_X_test)
#     accuracy, precision, sensitivity, specificity, false_discovery_rate, F1_score = perf_metrics(y_pred, Y_test)
#     accuracy_test.append(accuracy)
#     precision_test.append(precision)
#     sensitivity_test.append(sensitivity)
#     specificity_test.append(specificity)
#     false_discovery_rate_test.append(false_discovery_rate)
#     F1_score_test.append(F1_score)
#     print_stuffs(accuracy, sensitivity, specificity, precision, false_discovery_rate, F1_score)
    
#     print("\n-----------------\n")
    
# plt.plot(feature_counts, accuracy_train, label='accuracy_train')
# plt.plot(feature_counts, accuracy_test, label='accuracy_test')
# plt.plot(feature_counts, F1_score_train, label='F1_score_train')
# plt.plot(feature_counts, F1_score_test, label='F1_score_test')
# plt.xlabel('max_feature_count')
# plt.ylabel('performance metrics')
# plt.legend()
# plt.show()

    


    
    