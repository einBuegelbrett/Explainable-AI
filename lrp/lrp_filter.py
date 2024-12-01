import torch

def relevance_filter(r: torch.tensor, top_k_percent: float = 1.0) -> torch.tensor:
    assert 0.0 < top_k_percent <= 1.0

    if top_k_percent < 1.0:
        size = r.size()
        r = r.flatten(start_dim=1)
        num_elements = r.size(-1)
        k = max(1, int(top_k_percent * num_elements))
        # Get the top-k relevance values and their indices.
        top_k = torch.topk(input=r, k=k, dim=-1)
        # Create a tensor of zeros with the same shape as the flattened relevance tensor
        r = torch.zeros_like(r)
        # Scatter the top-k values into the new tensor at their original positions
        r.scatter_(dim=1, index=top_k.indices, src=top_k.values)
        # Reshape the filtered relevance scores to match the original tensor shape
        return r.view(size)
    else:
        # If top_k_percent is 1.0, return the relevance tensor unchanged
        return r