from typing import List, Union

import torch
from torch.nn import ParameterList, Parameter

from allennlp.common.checks import ConfigurationError


class ScalarMix(torch.nn.Module):
    """
    Computes a parameterised scalar mixture of N tensors, ``mixture = gamma * sum(s_k * tensor_k)``
    where ``s = softmax(w)``, with ``w`` and ``gamma`` scalar parameters.

    In addition, if ``do_layer_norm=True`` then apply layer normalization to each tensor
    before weighting.
    """

    def __init__(
            self,
            mixture_size: int,
            do_layer_norm: bool = False,
            initial_scalar_parameters: List[float] = None,
            trainable: bool = True,
    ) -> None:
        super().__init__()
        self.mixture_size = mixture_size
        self.do_layer_norm = do_layer_norm

        if initial_scalar_parameters is None:
            initial_scalar_parameters = [0.0] * mixture_size
        elif len(initial_scalar_parameters) != mixture_size:
            raise ConfigurationError(
                "Length of initial_scalar_parameters {} differs "
                "from mixture_size {}".format(initial_scalar_parameters, mixture_size)
            )

        self.scalar_parameters = ParameterList(
            [
                Parameter(
                    torch.FloatTensor([initial_scalar_parameters[i]]), requires_grad=trainable
                )
                for i in range(mixture_size)
            ]
        )
        self.gamma = Parameter(torch.FloatTensor([1.0]), requires_grad=trainable)

    def forward(self, tensors: List[torch.Tensor], mask: torch.Tensor = None) -> torch.Tensor:
        """
        Compute a weighted average of the ``tensors``.  The input tensors an be any shape
        with at least two dimensions, but must all be the same shape.

        When ``do_layer_norm=True``, the ``mask`` is required input.  If the ``tensors`` are
        dimensioned  ``(dim_0, ..., dim_{n-1}, dim_n)``, then the ``mask`` is dimensioned
        ``(dim_0, ..., dim_{n-1})``, as in the typical case with ``tensors`` of shape
        ``(batch_size, timesteps, dim)`` and ``mask`` of shape ``(batch_size, timesteps)``.

        When ``do_layer_norm=False`` the ``mask`` is ignored.
        """
        if len(tensors) != self.mixture_size:
            raise ConfigurationError(
                "{} tensors were passed, but the module was initialized to "
                "mix {} tensors.".format(len(tensors), self.mixture_size)
            )

        def _do_layer_norm(tensor, broadcast_mask, num_elements_not_masked):
            tensor_masked = tensor * broadcast_mask
            mean = torch.sum(tensor_masked) / num_elements_not_masked
            variance = (
                    torch.sum(((tensor_masked - mean) * broadcast_mask) ** 2) / num_elements_not_masked
            )
            return (tensor - mean) / torch.sqrt(variance + 1e-12)

        normed_weights = torch.nn.functional.softmax(
            torch.cat([parameter for parameter in self.scalar_parameters]), dim=0
        )
        normed_weights = torch.split(normed_weights, split_size_or_sections=1)

        if not self.do_layer_norm:
            pieces = []
            for weight, tensor in zip(normed_weights, tensors):
                pieces.append(weight * tensor)
            return self.gamma * sum(pieces)

        else:
            mask_float = mask.float()
            broadcast_mask = mask_float.unsqueeze(-1)
            input_dim = tensors[0].size(-1)
            num_elements_not_masked = torch.sum(mask_float) * input_dim

            pieces = []
            for weight, tensor in zip(normed_weights, tensors):
                pieces.append(
                    weight * _do_layer_norm(tensor, broadcast_mask, num_elements_not_masked)
                )
            return self.gamma * sum(pieces)


class SumMix(torch.nn.Module):
    def __init__(self, layers_to_sum: List, **kwargs):
        super().__init__()
        assert len(layers_to_sum) > 1
        self.indices_to_sum = layers_to_sum
        self.max_layer_idx = max(self.indices_to_sum)

    def forward(self, tensors: Union[List[torch.Tensor], torch.Tensor], *args, **kwargs):
        """
        computes the sum of the embeddings of the layers defined self.indices_to_sum
        :param tensors:  The input tensors can be any shape with at least two dimensions, but must all be the same shape.
        :return: the sum of the last dimension of tensors for the indices in the first dimension corresponding to self.indices_to_sum
        """
        if self.max_layer_idx > len(tensors):
            raise RuntimeError("The input indices has the index {} that is out of the bound of the input tensor "
                               "list with size {}".format(self.max_layer_idx, len(tensors)))
        return sum(tensors[self.indices_to_sum])
        # accum_tensor = tensors[self.indices_to_sum[0]]
        # for idx in self.indices_to_sum[1:]:
        #     accum_tensor = accum_tensor + tensors[idx]
        # return accum_tensor