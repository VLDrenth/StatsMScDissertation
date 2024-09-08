import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from enum import Enum

from torch.utils.data import DataLoader, Dataset
from sklearn.cluster import KMeans
from dataclasses import dataclass
from batchbald_redux.batchbald import CandidateBatch 
from batchbald_redux.consistent_mc_dropout import BayesianModule, freeze_encoder_context
from sklearn.metrics import pairwise_distances
from scipy import stats

##
## Copied from batchbald_redux which is based on Jordan Ash's implementation
##


class BayesianMNISTCNN_EBM(BayesianModule):
    """Without Softmax."""

    def __init__(self, num_classes=10):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        self.fc1 = nn.Linear(512, 32)
        self.fc2 = nn.Linear(32, num_classes)
        
        # move model to device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)


    def mc_forward_impl(self, input: torch.Tensor, freeze_encoder: bool):
        with freeze_encoder_context(freeze_encoder):
            input = F.relu(F.max_pool2d((self.conv1(input)), 2))
            input = F.relu(F.max_pool2d((self.conv2(input)), 2))
            input = input.view(-1, 512)
            input = F.relu((self.fc1(input)))

        embedding = input
        input = self.fc2(input)

        return input, embedding

class BayesianMLP_EBM(BayesianModule):
    """Without Softmax."""

    def __init__(self, num_classes=10):
        super().__init__()

        self.fc1 = nn.Linear(768, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, num_classes)
        
        # move model to device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def mc_forward_impl(self, input: torch.Tensor, freeze_encoder: bool):
        with freeze_encoder_context(freeze_encoder):
            input = F.relu((self.fc1(input)))
            input = F.relu((self.fc2(input)))

        embedding = input
        input = self.fc3(input)

        return input, embedding


class GradEmbeddingType(Enum):
    BIAS = 0
    LINEAR = 1
    BIAS_LINEAR = 2

    def get_grad_embedding(self, embedding_N_K_E, loss_grad_N_K_C):
        embedding_dim = embedding_N_K_E.shape[2]
        num_classes = loss_grad_N_K_C.shape[2]

        if self != GradEmbeddingType.BIAS:
            # TODO: this seems very inefficient. Could do the same via broadcasting!
            # loss_grad_expanded_N_K_EC = torch.repeat_interleave(loss_grad_N_K_C, embedding_dim, dim=2)
            # grad_embedding_N_K_EC = loss_grad_expanded_N_K_EC * embedding_N_K_E.repeat(2, num_classes)
            loss_grad_N_K_1_C = loss_grad_N_K_C[:, :, None, :]
            embedding_N_K_E_1 = embedding_N_K_E[:, :, :, None]
            grad_embedding_N_K_EC = (loss_grad_N_K_1_C * embedding_N_K_E_1).flatten(2)

        if self == GradEmbeddingType.BIAS:
            return loss_grad_N_K_C
        elif self == GradEmbeddingType.LINEAR:
            return grad_embedding_N_K_EC
        elif self == GradEmbeddingType.BIAS_LINEAR:
            return torch.cat([loss_grad_N_K_C, grad_embedding_N_K_EC], dim=2)
        else:
            raise NotImplementedError(f"Unknown GradEmbeddingType {self}!")

    def get_grad_embedding_shape(self, N, K, embedding_B_L_E, loss_grad_B_L_C):
        embedding_dim = embedding_B_L_E.shape[2]
        num_classes = loss_grad_B_L_C.shape[2]

        if self == GradEmbeddingType.BIAS:
            grad_embedding_shape = (N, K, num_classes)
        elif self == GradEmbeddingType.LINEAR:
            grad_embedding_shape = (N, K, embedding_dim * num_classes)
        elif self == GradEmbeddingType.BIAS_LINEAR:
            grad_embedding_shape = (N, K, (embedding_dim + 1) * num_classes)
        else:
            raise NotImplementedError(f"Unknown GradEmbeddingType {self}!")

        return grad_embedding_shape


def init_centers(X, K):
    ind = np.argmax([np.linalg.norm(s, 2) for s in X])
    mu = [X[ind]]
    indsAll = [ind]
    centInds = [0.] * len(X)
    cent = 0
    print('#Samps\tTotal Distance')
    while len(mu) < K:
        if len(mu) == 1:
            D2 = pairwise_distances(X, mu).ravel().astype(float)
        else:
            newD = pairwise_distances(X, [mu[-1]]).ravel().astype(float)
            for i in range(len(X)):
                if D2[i] > newD[i]:
                    centInds[i] = cent
                    D2[i] = newD[i]
        print(str(len(mu)) + '\t' + str(sum(D2)), flush=True)
        # if sum(D2) == 0.0:
        #    pdb.set_trace()
        D2 = D2.ravel().astype(float)
        Ddist = (D2 ** 2) / sum(D2 ** 2)
        customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
        ind = customDist.rvs(size=1)[0]
        while ind in indsAll: ind = customDist.rvs(size=1)[0]
        mu.append(X[ind])
        indsAll.append(ind)
        cent += 1
    return indsAll

@dataclass
class CandidateBatchComputer:
    acquisition_size: int

    def compute_candidate_batch(
        self, model, pool_loader: torch.utils.data.DataLoader, device
    ) -> CandidateBatch:
        pass

@dataclass
class BADGE(CandidateBatchComputer):
    def compute_candidate_batch(
        self, model, pool_loader: torch.utils.data.DataLoader, device
    ) -> CandidateBatch:
        grad_embeddings = model.get_grad_embeddings(pool_loader, num_samples=0, loss=torch.nn.functional.nll_loss,
                                                    model_labels=True, grad_embedding_type=GradEmbeddingType.LINEAR,
                                                    device=device, storage_device="cpu")
        chosen_indices = init_centers(grad_embeddings.squeeze(1).numpy(), self.acquisition_size)

        return CandidateBatch(indices=chosen_indices, scores=[0.0] * len(chosen_indices))


@dataclass
class TrainedModel:
    """Evaluate a trained model."""

    def get_log_probs_N_K_C_labels_N(
        self, loader: DataLoader, num_samples: int, device: object, storage_device: object
    ):
        raise NotImplementedError()

    def get_log_probs_N_K_C(self, loader: DataLoader, num_samples: int, device: object, storage_device: object):
        log_probs_N_K_C, labels = self.get_log_probs_N_K_C_labels_N(loader, num_samples, device, storage_device)
        return log_probs_N_K_C

    def get_grad_embeddings(
        self,
        loader: DataLoader,
        num_samples: int,
        loss,
        grad_embedding_type: GradEmbeddingType,
        model_labels: bool,
        device: object,
        storage_device: object,
    ):
        raise NotImplementedError()


@dataclass
class TrainedBayesianModel(TrainedModel):
    model: BayesianModule

    def get_log_probs_N_K_C_labels_N(
        self, loader: DataLoader, num_samples: int, device: object, storage_device: object
    ):
        log_probs_N_K_C, labels_B = self.model.get_predictions_labels(
            num_samples=num_samples, loader=loader, device=device, storage_device=storage_device
        )

        # NOTE: this wastes memory bandwidth, but is needed for ensembles where more than one model might not fit
        # into memory.
        #self.model.to("cpu")

        return log_probs_N_K_C, labels_B

    def get_grad_embeddings(
        self,
        loader: DataLoader,
        num_samples: int,
        loss,
        grad_embedding_type: GradEmbeddingType,
        model_labels: bool,
        device: object,
        storage_device: object,
    ):
        grad_embeddings_N_K_E = self.model.get_grad_embeddings(
            num_samples=num_samples,
            loader=loader,
            loss=loss,
            grad_embedding_type=grad_embedding_type,
            model_labels=model_labels,
            device=device,
            storage_device=storage_device,
        )
        return grad_embeddings_N_K_E

    def get_embeddings(
        self,
        loader: DataLoader,
        num_samples: int,
        device: object,
        storage_device: object,
    ):
        embeddings_N_K_E = self.model.get_grad_embeddings(
            num_samples=num_samples,
            loader=loader,
            device=device,
            storage_device=storage_device,
        )
        return embeddings_N_K_E


def badge_selection_custom(model, pool_dataset, batch_size):
    """
    DO NOT USE THIS FUNCTION. Only for experimentation purposes.


    Implement the BADGE selection strategy for a single iteration.
    
    Args:
    - model: PyTorch model
    - pool_dataset: Dataset containing unlabeled examples (U \ S)
    - batch_size: Number of examples to select (B)
    
    Returns:
    - indices: Indices of selected examples to be added to S
    """
    raise Warning("This function is only for experimentation purposes. Do not use it in the main code.")
    
    model.eval()
    gradient_embeddings = []
    
    # Compute gradient embeddings for all examples in U \ S
    for idx, (x, _) in enumerate(pool_dataset):
        x = x.unsqueeze(0)  # Add batch dimension
        
        # Forward pass
        output = model(x)
        
        # Compute hypothetical label
        y_hat = output.argmax(dim=1)
        
        # Compute gradient embedding
        loss = F.cross_entropy(output, y_hat)
        
        # Compute gradients w.r.t. the last layer parameters
        last_layer = list(model.parameters())[-2:]
        last_layer = (y for y in last_layer)
        grad_embedding = torch.autograd.grad(loss, 
                                             last_layer, create_graph=False)[0]
        
        gradient_embeddings.append(grad_embedding.cpu().detach().numpy().flatten())
    
    # Convert to numpy array
    gradient_embeddings = np.array(gradient_embeddings)
    
    # Use k-MEANS++ to select diverse samples
    kmeans = KMeans(n_clusters=batch_size, init='k-means++', n_init=1, max_iter=1)
    kmeans.fit(gradient_embeddings)
    
    # Get the indices closest to the centroids
    distances = kmeans.transform(gradient_embeddings)
    selected_indices = np.argmin(distances, axis=0)
    
    return selected_indices