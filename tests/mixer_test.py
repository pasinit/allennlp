from allennlp.modules.scalar_mix import SumMix
import torch

if __name__ == "__main__":
    indices = [-1, -2]
    mixer = SumMix([-1, -2])
    tensors = [torch.randn([32, 10, 1024])] * 12
    merged = mixer(tensors)
    test_tensor = tensors[-1] + tensors[-2]
    assert torch.sum(test_tensor != merged).item() == 0