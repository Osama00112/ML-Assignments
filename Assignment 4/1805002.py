import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy import linalg as la
import seaborn as sns
import scipy.stats as stats


dataset1 = 'data/2D_data_points_1.txt'
dataset2 = 'data/2D_data_points_2.txt'
dataset3 = 'data/3D_data_points.txt'
dataset4 = 'data/6D_data_points.txt'



np.random.seed(1)

dataset_path = dataset4
plot_path = 'plots/dataset4/'

df = pd.read_csv(dataset_path, header=None)

def pca(dataframe, dim):
    df = dataframe.copy()
    df_norm = (df - df.mean()) / df.std()   
    cov_mat = np.cov(df_norm.T)
    U, S, Vt = la.svd(cov_mat)
    
    reduced_data = np.dot(df_norm, Vt[:dim,:].T)
    
    return reduced_data


reduced_data = pca(df, 2)

# plot the data along the 2 axis
plt.scatter(reduced_data[:, 0], reduced_data[:, 1])
plt.savefig('plots/pca.png')
plt.show()

def E_step(data, mu, sigma, pi):
    """
    Args:
        data : reduced to 2 dim datapoints
        mu : mu matrix of shape (k,2)
        sigma : sigma matrix of shape (k,2,2)
        pi : pi matrix of shape (k,)
    Returns:
        likelihood_matrix : matrix of shape (n,k) where n is the number of datapoints
    """
    
    n = data.shape[0]
    k = mu.shape[0]
    likelihood_matrix = np.zeros((n, k))
    
    for i in range(k):
        likelihood_matrix[:, i] = pi[i] * stats.multivariate_normal.pdf(data, mu[i], sigma[i])
        
    # normalize the likelihood matrix
    likelihood_matrix = likelihood_matrix / np.sum(likelihood_matrix, axis=1).reshape(-1, 1)
    
    return likelihood_matrix    


def M_step(likelihood_matrix, data):
    """
    Args:
        likelihood_matrix : pij matrix of shape (n,k)
    Returns:
        mu : mu matrix of shape (k,2)
        sigma : sigma matrix of shape (k,2,2)
        pi : pi matrix of shape (k,)
    """
    
    k = likelihood_matrix.shape[1]
    
    n = np.sum(likelihood_matrix, axis=0)
    mu = np.zeros((k, 2))
    sigma = np.zeros((k, 2, 2))
    pi = np.zeros(k)
    
    for i in range(k):
        mu[i] = np.dot(likelihood_matrix[:, i], data) / n[i]
        
        sigma[i] = np.dot(likelihood_matrix[:, i] * (data - mu[i]).T, data - mu[i]) / n[i]
        sigma[i] += 1e-6 * np.eye(2)  # regularization term for avoiding singularity
        
        pi[i] = n[i] / np.sum(n)
    
    return mu, sigma, pi    

def initialization(num_clusters, dim, data):
    k = num_clusters
    n = data.shape[0]
    mu = np.random.rand(k, dim)
    sigma = np.array([np.eye(dim)] * k)
    pi = np.ones(k) / k
    
    return mu, sigma, pi
    
def calculate_log_likelihood(data, k, mu, sigma, pi):
    n = data.shape[0]
    log_likelihood = np.zeros((n, k))
    
    for i in range(k):
        log_likelihood[:, i] = pi[i] * stats.multivariate_normal.pdf(data, mu[i], sigma[i])
    
    # normalize
    log_likelihood = log_likelihood / np.sum(log_likelihood, axis=1).reshape(-1, 1)  
    log_likelihood = np.sum(np.log(np.sum(log_likelihood, axis=1)))
    
    return log_likelihood

def EM_algorithm(data, k, trials=5, epoch=2000, tol=1e-6):
    n = data.shape[0]
    dim = data.shape[1]
    
    best_log_likelihood = -np.inf
    best_mu = None
    best_sigma = None
    best_pi = None
    best_iteration = None
    
    
    likelihood_matrix = np.zeros((n, k))
    log_likelihood_list = []
    
    
    for t in range(trials):
        
        prev_log_likelihood = None
        # Random initialization of parameters
        mu, sigma, pi = initialization(k, dim, data)
        
        for iteration in range(epoch):
            # E-step
            likelihood_matrix = E_step(data, mu, sigma, pi)
            
            # M-step
            new_mu, new_sigma, new_pi = M_step(likelihood_matrix, data)
            
            # log likelihood
            log_likelihood = calculate_log_likelihood(data, k, new_mu, new_sigma, new_pi)
            
            # check for convergence
            if np.abs(np.sum(new_mu - mu)) < tol and np.abs(np.sum(new_sigma - sigma)) < tol and np.abs(np.sum(new_pi - pi)) < tol:
                break
            
            prev_log_likelihood = log_likelihood
            
            mu = new_mu
            sigma = new_sigma
            pi = new_pi
        
        if log_likelihood > best_log_likelihood:
            best_log_likelihood = log_likelihood
            best_mu = mu
            best_sigma = sigma
            best_pi = pi
            best_iteration = iteration
        
    return best_mu, best_sigma, best_pi, best_iteration, best_log_likelihood

"""
trials = 5
epoch = 500
k_values = [3, 4, 5, 6, 7, 8]
best_log_values = []
best_k_value = None
best_parameters = None  
best_log_likelihood = -np.inf   

for k in k_values:
    mu, sigma, pi, iterations, log_likeihood = EM_algorithm(reduced_data, k=k, trials=trials, epoch=epoch)
    best_log_values.append(log_likeihood)
    print('k = {}, log likelihood = {}, iterations = {}'.format(k, log_likeihood, iterations))
    if log_likeihood > best_log_likelihood:
        best_log_likelihood = log_likeihood
        best_parameters = (mu, sigma, pi)
        best_k_value = k

print('Best k = {}'.format(best_k_value))
print('Best log likelihood = {}'.format(best_log_likelihood))
    
plt.plot(k_values, best_log_values)
plt.xlabel('No. of clusters')
plt.ylabel('log likelihood')    
    
"""




