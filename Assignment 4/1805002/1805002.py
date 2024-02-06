import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy import linalg as la
import seaborn as sns
import scipy.stats as stats
from matplotlib.patches import Ellipse
import matplotlib
import imageio
import io
#listedcolormap
from matplotlib.colors import ListedColormap

dataset_number = input("Enter dataset number: ")

datasets = {
    '1': 'data/2D_data_points_1.txt',
    '2': 'data/2D_data_points_2.txt',
    '3': 'data/3D_data_points.txt',
    '4': 'data/6D_data_points.txt',
    '5': 'data/100d.txt'
}

dataset_path = datasets[dataset_number]
plot_path = f'plots/dataset{dataset_number}/'


# dataset1 = 'data/2D_data_points_1.txt'
# dataset2 = 'data/2D_data_points_2.txt'
# dataset3 = 'data/3D_data_points.txt'
# dataset4 = 'data/6D_data_points.txt'

np.random.seed(1)

# dataset_path = dataset4
# plot_path = 'plots/dataset4/'
df = pd.read_csv(dataset_path, header=None)

#df.head()
# check the scales of the features
df.describe()


def pca(dataframe, dim):
    df = dataframe.copy()
    df_norm = (df - df.mean()) / df.std()   
    cov_mat = np.cov(df_norm.T)
    U, S, Vt = la.svd(cov_mat)
    
    reduced_data = np.dot(df_norm, Vt[:dim,:].T)
    
    return reduced_data

# ## Source: pca+svd intuition:
# https://stats.stackexchange.com/questions/134282/relationship-between-svd-and-pca-how-to-use-svd-to-perform-pca#:~:text=Principal%20component%20analysis%20(PCA)%20is,of%20the%20data%20matrix%20X.
# 
# ## Source: Eigen values are in decreasing order
# https://www.youtube.com/watch?v=P5mlg91as1c&ab_channel=ArtificialIntelligence-AllinOne


# if feature count = 2, dont pca
if df.shape[1] == 2:
    reduced_data = df.values
    # normlaize
    reduced_data = (reduced_data - reduced_data.mean()) / reduced_data.std()
else:
    reduced_data = pca(df, 2)

# convert to dataframe
df_reduced = pd.DataFrame(reduced_data, columns=['PC1', 'PC2'])
df_reduced.describe()


# plot the data along the 2 axis
plt.scatter(reduced_data[:, 0], reduced_data[:, 1])
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA')
#plt.savefig(plot_path + 'pca.png')
plt.show()


def E_step(data, mu, sigma, pi):
    # likelihood of data point given the cluster
    """_summary_

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
    
    # n_i
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
    #log_likelihood = log_likelihood / np.sum(log_likelihood, axis=1).reshape(-1, 1)  
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


# check em algo 
# k = 3
# trials = 5
# epoch = 200
# mu, sigma, pi, iterations, best_log_likihood = EM_algorithm(reduced_data, k=k, trials=trials, epoch=epoch)
# # check the log likelihood
# print(best_log_likihood)
# print(iterations)
# # plot the clusters, each cluster is represented by a different color for k = 5
# plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=np.argmax(E_step(reduced_data, mu, sigma, pi), axis=1))
# #plt.savefig('plots/em.png')
# plt.show()

def check_for_specific_k(data, k, trials=5, epoch=200):
    mu, sigma, pi, iterations, best_log_likihood = EM_algorithm(data, k=k, trials=trials, epoch=epoch)
    print('k = {}, log likelihood = {}, iterations = {}'.format(k, best_log_likihood, iterations))
    plt.scatter(data[:, 0], data[:, 1], c=np.argmax(E_step(data, mu, sigma, pi), axis=1))
    plt.show()
            

# Function to draw ellipse
def draw_ellipse(mu, sigma, ax, color, weights=None, **kwargs):
    if weights is None:
        weights = [1] * len(mu)
    for i in range(len(mu)):
        # set color for each cluster
        kwargs['color'] = color[i]
        U, s, Vt = la.svd(sigma[i])
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
        for nsig in range(1, 4):
            ax.add_patch(Ellipse(mu[i], nsig * width, nsig * height, angle=angle, **kwargs))
            

#check_for_specific_k(reduced_data, k=3, trials=5, epoch=200)

k_values_input = input('Enter k values: ')
k_values = list(map(int, k_values_input.split()))
trials = input('Enter no. of trials: ')
trials = int(trials)
epoch = input('Enter no. of epochs: ')
epoch = int(epoch)

# trials = 5
# epoch = 500
# k_values = [3, 4, 5, 6, 7, 8]
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
        
    fig, ax = plt.subplots()
    cluster_assignments = np.argmax(E_step(reduced_data, mu, sigma, pi), axis=1)
    #cmap = plt.cm.get_cmap('viridis', len(mu))
    cmap = ListedColormap(sns.color_palette("bright", len(mu)))


    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=cluster_assignments, cmap=cmap, s=5)

    draw_ellipse(mu, sigma, ax, color=[cmap(i) for i in range(len(mu))], weights=pi, alpha=0.6)
    
    plt.title('GMM-EM, k = {}'.format(k))
    plt.xlabel('feature 1')
    plt.ylabel('feature 2')
    plt.savefig(plot_path + 'new_em_{}.png'.format(k))
    plt.show()   
    

print('Best k = {}'.format(best_k_value))
print('Best log likelihood = {}'.format(best_log_likelihood))    


#clear plot    
plt.plot(k_values, best_log_values)
plt.xlabel('No. of clusters')
plt.ylabel('log likelihood')
plt.title('log likelihood vs k')
plt.grid()
plt.savefig(plot_path + 'new_log_likelihood_vs_k.png')
plt.show()

# ## Making Animation
def EM_algorithm_create_gif(data, k, trials=5, epoch=2000, tol=1e-6, fps=4):
    n = data.shape[0]
    dim = data.shape[1]

    best_log_likelihood = -np.inf
    best_mu = None
    best_sigma = None
    best_pi = None
    best_iteration = None

    likelihood_matrix = np.zeros((n, k))
    log_likelihood_list = []

    # Create a colormap for the clusters
    #cmap = plt.cm.get_cmap('viridis', k)
    cmap = ListedColormap(sns.color_palette("bright", k))

    fig, ax = plt.subplots()
    
    gif_images = []
    
    for trials in range(1, trials + 1):

        mu, sigma, pi = initialization(k, dim, data)

        for iter in range(1, epoch + 1):
            # E-step
            likelihood_matrix = E_step(data, mu, sigma, pi)

            # M-step
            new_mu, new_sigma, new_pi = M_step(likelihood_matrix, data)

            # log likelihood
            log_likelihood = calculate_log_likelihood(data, k, new_mu, new_sigma, new_pi)

            # check for convergence
            if np.abs(np.sum(new_mu - mu)) < tol and np.abs(np.sum(new_sigma - sigma)) < tol and np.abs(
                    np.sum(new_pi - pi)) < tol:
                break

            mu = new_mu
            sigma = new_sigma
            pi = new_pi

            if iter % 1 == 0:
                # Clear previous plot and draw new one
                ax.clear()

                # Plot scatter points with cluster colors
                cluster_assignments = np.argmax(likelihood_matrix, axis=1)
                plt.scatter(data[:, 0], data[:, 1], c=cluster_assignments, cmap=cmap, s=1)

                # Draw ellipses with matching colors
                draw_ellipse(mu, sigma, ax, color=[cmap(i) for i in range(len(mu))], weights=pi, alpha=0.6)
                
                # Set x and y limits explicitly
                plt.xlim(min(data[:, 0]), max(data[:, 0]))
                plt.ylim(min(data[:, 1]), max(data[:, 1]))
                
                # Add title to the plot
                plt.title(f'GMM-EM {k} Gaussian, Iteration: {iter}, Trial: {trials}')
                # Set x and y limits explicitly
                plt.xlim(min(data[:, 0]), max(data[:, 0]))
                plt.ylim(min(data[:, 1]), max(data[:, 1]))
                
                plt.xlabel('x')
                plt.ylabel('y')
                #plt.legend()

                # Convert plot to image
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                gif_images.append(imageio.v2.imread(buf))
                
        if log_likelihood > best_log_likelihood:
            best_log_likelihood = log_likelihood
            best_mu = mu
            best_sigma = sigma
            best_pi = pi
            best_iteration = iter

    # Save GIF
    imageio.mimsave(plot_path + 'new_em_animation.gif', gif_images, fps=fps)

    return best_mu, best_sigma, best_pi, best_iteration, best_log_likelihood



k = input('Enter no. of clusters for GMM-EM animation: ')
k = int(k)
trials = input('Enter no. of trials: ')
trials = int(trials)
epoch = input('Enter no. of epochs: ')
epoch = int(epoch)
fps = input('Enter fps: ')
fps = int(fps)


best_mu, best_sigma, best_pi, iteration, _ = EM_algorithm_create_gif(reduced_data, k=k, trials=trials, epoch=epoch, fps=fps)

# plot for the best parameters
fig, ax = plt.subplots()
cluster_assignments = np.argmax(E_step(reduced_data, best_mu, best_sigma, best_pi), axis=1)
#cmap = plt.cm.get_cmap('viridis', len(best_mu))
cmap = ListedColormap(sns.color_palette("bright", len(best_mu)))
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=cluster_assignments, cmap=cmap, s=5)
draw_ellipse(best_mu, best_sigma, ax, color=[cmap(i) for i in range(len(best_mu))], weights=best_pi, alpha=0.6)
plt.show()

