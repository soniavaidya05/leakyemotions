from typing import Callable, Sequence, TypeVar, Union
import torch
from torch import nn


encoder = TypeVar("encoder")


def contrastive_predictive_coding(
    state: torch.Tensor,
    action: torch.Tensor,
    next_states: torch.Tensor,
    negative_states: torch.Tensor,
    encoder: nn.Module,
    prediction_head: nn.Module,
) -> torch.Tensor:
    latent_state = encoder(state)
    latent_next_state = encoder(next_states)
    latent_negative_states = encoder(negative_states)

    predicted_next_state = prediction_head(latent_state, action)

    # infonce loss
    positive_dot_product = torch.sum(latent_next_state * predicted_next_state, dim=1)
    negative_dot_product = torch.sum(
        latent_negative_states * predicted_next_state, dim=1
    )
    positive_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        positive_dot_product, torch.ones_like(positive_dot_product)
    )
    negative_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        negative_dot_product, torch.zeros_like(negative_dot_product)
    )
    return positive_loss + negative_loss


def cosine_self_predictive_coding(
    state: torch.Tensor,
    action: torch.Tensor,
    next_states: torch.Tensor,
    encoder: nn.Module,
    prediction_head: nn.Module,
) -> torch.Tensor:
    latent_state = encoder(state)
    latent_next_state = encoder(next_states)

    predicted_next_state = prediction_head(latent_state, action)

    # infonce loss
    cosine_sim = torch.nn.functional.cosine_similarity(
        latent_next_state, predicted_next_state, dim=1
    )
    return cosine_sim


def mse_self_predictive_coding(
    state: torch.Tensor,
    action: torch.Tensor,
    next_states: torch.Tensor,
    encoder: nn.Module,
    prediction_head: nn.Module,
) -> torch.Tensor:
    latent_state = encoder(state)
    latent_next_state = encoder(next_states)

    predicted_next_state = prediction_head(latent_state, action)

    # infonce loss
    mse = torch.nn.functional.mse_loss(latent_next_state, predicted_next_state)
    return mse


# # usage examples
# if __name__ == "__main__":
#     # assume tensors are indexed (B,t,x), where B is the batch dimension, t is a temporal dimension and
#     # x is the feature vector
#     # k is the number of time steps we sample the contrastive (or noncontrastive) pair from
#     # the assumption is that our encode takes a state
#     # and the prediction head takes a state and an action sequence. The dependency on the action sequence can be removed,
#     # but then there are mathematical problems with the formulations due to necessary off-policy corrections
#     k = ???
#     next_states = next_states[:, k]
#     negative_idx = torch.randint(len(next_states), (len(next_states),))
#     negatives = next_states[negative_idx]
#
#     contrastive_predictive_coding(state, actions, next_states, negatives, encoder, prediction_head)
