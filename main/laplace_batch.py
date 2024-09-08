import torch
import numpy as np
from scipy.stats import dirichlet
from .badge import BADGE, TrainedBayesianModel, init_centers
from tqdm.auto import tqdm
from batchbald_redux.batchbald import CandidateBatch
from .bald_sampling import compute_bald, compute_entropy, compute_emp_cov, max_joint_eig
from .approximation import compute_S, compute_phi_S
import math


def get_laplace_batch(model, pool_loader, acquisition_batch_size, method, device=None):
    '''
    model: Laplace model
    acquisition_batch_size: how many observations to return
    device: device to run on
    pool_loader: data loader for the pool
    method: method to use for batch selection

    Returns: batch of observations (CandiateBatch object)
    '''
    scores = torch.empty(len(pool_loader.dataset), 1).to(device=device)
    scores.fill_(float('-inf'))

    if method == 'random':
        indices = torch.randperm(len(pool_loader.dataset))[:acquisition_batch_size]
        return CandidateBatch(indices=indices.tolist(), scores=[0]*acquisition_batch_size)    
    elif method == 'entropy':
        for i, (data, _) in tqdm(enumerate(pool_loader), desc="Computing entropies", leave=True): 
            data = data.to(device=device)
            scoring_batch_size = data.shape[0]

            ent = compute_entropy(model, data)
            scores[i*scoring_batch_size:(i+1)*scoring_batch_size] = ent.unsqueeze(-1)

    elif method == 'bald':
        for i, (data, _) in tqdm(enumerate(pool_loader), desc="Computing BALD scores", leave=True): 
            data = data.to(device=device)
            scoring_batch_size = data.shape[0]

            # compute BALD
            bald = compute_bald(model, data, train_loader=None, refit=False, n_samples=10).unsqueeze(-1)

            scores[i*scoring_batch_size:(i+1)*scoring_batch_size] = bald
    elif method == 'max_logdet_S':
        # extract data from the pool
        pool_data = torch.cat([data for data, _ in pool_loader], dim=0).to(device=device)

        S = compute_S(model, pool_data)
        # add identity matrix to S
        mat = S + torch.eye(S.shape[0]).to(device=device)
        indices, log_det, _ = stochastic_greedy_maxlogdet(mat, acquisition_batch_size)
        return CandidateBatch(indices=indices, scores=[log_det]*acquisition_batch_size)
    elif method == 'badge':
        trained_model = TrainedBayesianModel(model)
        badge_selector = BADGE(acquisition_size=acquisition_batch_size)
        return badge_selector.compute_candidate_batch(trained_model, pool_loader, device)
    elif method == 'empirical_covariance':
         # extract data from the pool
        pool_data = torch.cat([data for data, _ in pool_loader], dim=0).to(device=device)

        # compute the empirical covariance matrix
        cov = compute_emp_cov(model, pool_data)

        # add identity matrix to S
        mat = cov + torch.eye(cov.shape[0]).to(device=device)

        indices, log_det, _ = stochastic_greedy_maxlogdet(mat, acquisition_batch_size)
        return CandidateBatch(indices=indices, scores=[log_det]*acquisition_batch_size)
    elif method == 'joint_eig':
        # extract data from the pool
        pool_data = torch.cat([data for data, _ in pool_loader], dim=0).to(device=device)

        indices, score = max_joint_eig(model=model, data=pool_data, K=1000, batch_size=acquisition_batch_size)

        return CandidateBatch(indices=indices, scores=[score]*acquisition_batch_size)
    elif method == 'similarity_kmeans':
        # extract data from the pool
        pool_data = torch.cat([data for data, _ in pool_loader], dim=0).to(device=device)
        mat = compute_phi_S(model, pool_data).cpu().numpy().astype(np.float32)

        # apply kmeans++ initalization to the normalized gradient embeddings
        indices = init_centers(mat, acquisition_batch_size)

        return CandidateBatch(indices=indices, scores=[0]*acquisition_batch_size)
    else:
        raise ValueError('Invalid method')
    
    # Compute top k scores
    values, indices = torch.topk(scores, acquisition_batch_size, largest=True, sorted=False, dim=0)
    return CandidateBatch(indices=indices.squeeze().tolist(), scores=values.squeeze().tolist())

def greedy_max_logdet(matrix, k):
    """
    Greedily selects k rows and columns from the input matrix to maximize the determinant.
    
    Args:
    matrix (torch.Tensor): NxN input matrix
    k (int): Size of the submatrix to select
    
    Returns:
    tuple: (selected_indices, max_determinant)
    """
    N = matrix.shape[0]
    if k > N:
        raise ValueError("k cannot be larger than the matrix size")
    
    # Initialize the list of selected indices
    selected_indices = []
    
    for _ in range(k):
        max_det = float('-inf')
        best_index = -1
        
        # Try adding each remaining index and calculate the determinant
        for i in range(N):
            if i not in selected_indices:
                current_indices = selected_indices + [i]
                submatrix = matrix[current_indices][:, current_indices]
                det = torch.det(submatrix).item()
                
                # Update if we found a better determinant
                if det > max_det:
                    max_det = det
                    best_index = i
        
        # Add the best index found in this iteration
        selected_indices.append(best_index)
    
    # Calculate the final determinant
    final_submatrix = matrix[selected_indices][:, selected_indices]
    max_determinant = torch.logdet(final_submatrix).item()/2
    
    return selected_indices, max_determinant, final_submatrix

def stochastic_greedy_maxlogdet(matrix, k, eps=0.2):
    """
    Stochastically selects k rows and columns from the input matrix to maximize the determinant.
    
    Args:
    matrix (torch.Tensor): NxN input matrix
    k (int): Size of the submatrix to select
    eps (int): Measure of how close to the optimal solution we want to be
    
    Returns:
    tuple: (selected_indices, max_determinant)
    """
    N = matrix.shape[0]
    if k > N:
        raise ValueError("k cannot be larger than the matrix size")
    
    # using formula from Lazier than Lazy Greedy 
    n_samples = int((N / k) * math.log(1/eps))

    # Initialize the list of selected indices
    selected_indices = []
    
    for _ in range(k):
        max_det = float('-inf')
        best_index = -1

        # take random subsample (of n_samples) from the remaining indices
        remaining_indices = [i for i in range(N) if i not in selected_indices]
        subsample = np.random.choice(remaining_indices, n_samples, replace=False)

        # Try adding each index from subsample and calculate the determinant
        for i in subsample:
            current_indices = selected_indices + [i]
            submatrix = matrix[current_indices][:, current_indices]
            
            det = torch.logdet(submatrix).item()

            # Update if we found a better determinant
            if det > max_det:
                max_det = det
                best_index = i
        
        # Add the best index found in this iteration
        selected_indices.append(best_index)
    
    # Calculate the final determinant
    final_submatrix = matrix[selected_indices][:, selected_indices]
    max_determinant = torch.logdet(final_submatrix).item()/2

    return selected_indices, max_determinant, final_submatrix