import torch
import numpy as np

def compute_entropy(la_model, data):
    # for each datapoint compute the probabilties of each class
    p = la_model(data, pred_type='glm', link_approx='probit')  # (n_data, n_classes)
    entropy = _h(p)

    return entropy

def _h(p):
    # p is a tensor of shape (n_data, n_classes)
    return -torch.sum(p * torch.log(p + 1e-16), dim=1)

def compute_multivariate_entropy(la_model, data):
    ent = compute_entropy(la_model, data)
    return ent.sum()

def compute_entropy_weights(la_model, data, n_samples=50):
    # Sample from the posterior
    posterior_weights = la_model.sample(n_samples=n_samples)
    probs = torch.zeros(n_samples, data.shape[0], 10)

    # Compute the entropy for each sample
    for i, weights in enumerate(posterior_weights):
        # Set the weights in the model
        if la_model.backend.last_layer:
            set_last_linear_layer_combined(la_model.model, weights)
        else:
            set_full_parameters(la_model.model, weights)

        # Compute the predictive distribution
        probs[i] = la_model(data, pred_type='glm', link_approx='probit')
    
    # Compute the entropy
    entropies = _h(probs.mean(dim=0))
    return entropies

def set_last_linear_layer_combined(model, new_weights_and_bias):
    # Find the last linear layer
    last_linear_layer = None
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            last_linear_layer = module
    
    if last_linear_layer is None:
        raise ValueError("No linear layer found in the model")

    # Get the shapes
    out_features, in_features = last_linear_layer.weight.shape
    
    # Check if the input tensor has the correct shape
    expected_shape = (out_features * in_features + out_features,)
    if new_weights_and_bias.shape != expected_shape:
        raise ValueError(f"Input tensor shape {new_weights_and_bias.shape} doesn't match the expected shape {expected_shape}")

    # Split the input tensor into weights and bias
    new_weights = new_weights_and_bias[:out_features * in_features].reshape(out_features, in_features)
    new_bias = new_weights_and_bias[out_features * in_features:]

    # Set new weights and bias
    last_linear_layer.weight.data = new_weights
    last_linear_layer.bias.data = new_bias

    return last_linear_layer

def compute_conditional_entropy(la_model, data, train_loader, refit=False, n_samples=50):
    # Sample from the posterior
    posterior_weights = la_model.sample(n_samples=n_samples)
    entropies = torch.zeros(posterior_weights.shape[0], data.shape[0])

    # Compute the entropy for each sample
    for i, weights in enumerate(posterior_weights):
        # Set the weights in the model
        if la_model.backend.last_layer:
            set_last_linear_layer_combined(la_model.model, weights)
        else:
            raise NotImplemented 

        if refit:
            # fit the model
            la_model.fit(train_loader)

            # Optimise the prior precision
            la_model.optimize_prior_precision(pred_type='glm', method='marglik', link_approx='probit', verbose=False)

        # Compute the predictive distribution
        probs = la_model(data, pred_type='glm', link_approx='probit')

        # Compute the entropy
        entropies[i] = _h(probs)

    return entropies.mean(dim=0)

def compute_bald(la_model, data, train_loader, refit=True, n_samples=50):
    # Compute the entropy
    entropy = compute_entropy(la_model, data)

    # Compute the conditional entropy
    cond_entropy = compute_conditional_entropy(la_model, data, train_loader, n_samples=n_samples, refit=refit)

    bald = entropy - cond_entropy

    return bald

def compute_normal_entropy(cov):
    return 0.5 * torch.logdet(cov) + 0.5 * cov.shape[0] * (1 + torch.log(torch.tensor(2 * torch.pi)))

def max_joint_eig(model, data, K, batch_size, eps=0.1):
    '''
    Function to greedily compute the maximum joint expected information gain
    --------------------------------
    Input:
    model: the laplace approximation model
    data: pool dataset
    K: the number of samples to compute the joint expected information gain
    --------------------------------
    
    Output:
    indices, eig: the maximum joint expected information gain and the indices of the selected samples    
    '''
    selected_indices = []
    N = data.shape[0]
    n_samples = int((N / batch_size) * np.log(1/eps))

    for _ in range(batch_size):
        max_eig = -torch.inf
        max_index = None

        # take random subsample (of n_samples) from the remaining indices
        remaining_indices = [i for i in range(N) if i not in selected_indices]
        subsample = np.random.choice(remaining_indices, n_samples, replace=False)

        for i in subsample:
            current_indices = selected_indices + [i]
            current_data = data[current_indices]

            eig = compute_joint_eig(model, current_data, K)
            print('eig:', eig)

            if max_index is None or eig > max_eig :
                max_eig = eig
                max_index = i
            
        selected_indices.append(max_index)

    return selected_indices, max_eig

def compute_joint_eig(model, x, K, C=10):
    '''
    Function to compute the joint expected information gain
    --------------------------------
    Input:
    model: the model
    x: the input data
    K: the number of samples to compute the joint expected information gain
    --------------------------------
    
    Output:
    eig: the joint expected information gain

    Details: Computes I[theta; Y | X] = H[Y|X] + H[THETA | X] - H[Y, theta | X]
    '''
    N = x.shape[0]
    C -= 1

    # Joint of Y and theta is given by p(y | theta, x) * p(theta | x) (Both conditional on training data)
    cov_joint = compute_joint_covariance(model, x, K)

    cov_y = cov_joint[:(N * C), :(N * C)]
    cov_theta = model.posterior_precision.to_matrix().inverse()

    det_y = torch.logdet(cov_y)
    det_theta = torch.logdet(cov_theta)
    det_joint = torch.logdet(cov_joint)

    eig = 0.5*(det_y + det_theta - det_joint)
    
    print('det_y:', det_y, 'det_theta:', det_theta, 'det_joint:', det_joint, 'eig:', eig)

    return eig

def compute_joint_covariance(model, x, K):
    
    N = x.shape[0]

    # Sample theta from the posterior
    posterior_weights = model.sample(n_samples=K)

    D = posterior_weights.shape[1]

    if K < N + D + 1:# Necessary condition for full rank covariance matrix
        # add warning
        print("Warning: K < N + D + 1. The covariance matrix may not be full rank")
        

    # For each theta, compute the predicted probabilities
    probs = torch.zeros(K, N, 9)

    for i, weights in enumerate(posterior_weights):
        set_last_linear_layer_combined(model.model, weights)
        probs[i] = model(x, pred_type='glm', link_approx='probit')[:, :-1]
    
    # Without flattening stack the probabilities and the weights into a tensor of shape: (K, (D + N))
    probs_and_theta = torch.cat([probs.view(K, -1), posterior_weights], dim=1)

    # reshape to (D + N, K)
    probs_and_theta = probs_and_theta.T
    
    # Obtain the covariance matrix of the joint distribution of Y and theta of shape: (D + N, D + N)
    cov_joint = torch.cov(probs_and_theta)

    return cov_joint


def compute_emp_cov(la, x_test, K=1000):
    res = la.predictive_samples(x_test, n_samples=K)  # shape K x N x C
    res  = res - res.mean(dim=0).unsqueeze(0)  # shape K x N x C

    # get probababilities of each class for each sample and each data point
    #probs = la(x_test).unsqueeze(0)  # shape N x C
    #res = res * probs  # shape K x N x C

    cov_k = torch.einsum('knc,kmc->nm', res, res) / K
    return cov_k
    

if __name__ == '__main__':
    pass


