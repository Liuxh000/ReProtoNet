import torch
from torch import Tensor, nn

def cosine_similarity(samples, prototypes) -> Tensor:
    """
    Compute prediction logits from their cosine distance to support set prototypes.
    Args:
        samples: features of the items to classify of shape (n_samples, feature_dimension)
    Returns:
        prediction logits of shape (n_samples, n_classes)
    """
    return (
            nn.functional.normalize(samples, dim=1)
            @ nn.functional.normalize(prototypes, dim=1).T
    )

def l2_distance_to_prototypes(samples, prototypes) -> Tensor:
    """
    Compute prediction logits from their euclidean distance to support set prototypes.
    Args:
        samples: features of the items to classify of shape (n_samples, feature_dimension)
    Returns:
        prediction logits of shape (n_samples, n_classes)
    """
    return -torch.cdist(samples, prototypes)

def get_grad_param(model_param):
    grad_param = dict()
    for key, param in model_param:
        if param.requires_grad:
            grad_param[key] = param
    return grad_param

def entropy(logits: Tensor) -> Tensor:
    """
    Compute entropy of prediction.
    WARNING: takes logit as input, not probability.
    Args:
        logits: shape (n_images, n_way)
    Returns:
        Tensor: shape(), Mean entropy.
    """
    probabilities = logits.softmax(dim=1)
    return (-(probabilities * (probabilities + 1e-12).log()).sum(dim=1)).mean()


def prototypical_loss(feature, label, n_shot):

    def supp_idxs(c):
        return label.eq(c).nonzero()[:n_shot].squeeze(1)
    labels = torch.unique(label)
    n_way = len(labels)
    support_idx = list(map(supp_idxs, labels))
    prototypes = torch.stack([feature[idx_list].mean(0) for idx_list in support_idx])

    query_idx = torch.stack(list(map(lambda c: label.eq(c).nonzero()[n_shot:], labels))).view(-1)
    query_samples = feature[query_idx]
    query_labels = label[query_idx]
    dists = l2_distance_to_prototypes(query_samples, prototypes)

    loss = nn.functional.cross_entropy(dists, query_labels)
    predict = dists.argmax(1)
    acc = sum(predict == query_labels)/len(query_labels)

    return loss, acc