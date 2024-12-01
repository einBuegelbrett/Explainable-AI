import torch
from torch import nn
from lrp.lrp_filter import relevance_filter

class RelevancePropagationConv2d(nn.Module):
    # Layer-wise relevance propagation for 2D convolution using the z^+-rule
    def __init__(
        self,
        layer: torch.nn.Conv2d,
        eps: float = 1.0e-05,
        top_k: float = 0.0,
    ) -> None:
        super().__init__()

        self.layer = layer

        # Apply the z^+ rule: Keep only non-negative weights for relevance propagation
        self.layer.weight = torch.nn.Parameter(self.layer.weight.clamp(min=0.0))
        # Set biases to zero to focus on weighted input contributions
        self.layer.bias = torch.nn.Parameter(torch.zeros_like(self.layer.bias))

        self.eps = eps
        self.top_k = top_k

    def forward(self, a: torch.tensor, r: torch.tensor) -> torch.tensor:
        # Filter the top-k% most relevant activations
        if self.top_k:
            r = relevance_filter(r, top_k_percent=self.top_k)
        # Compute activations for the current layer and add stabilization term eps
        z = self.layer.forward(a) + self.eps
        s = (r / z).data
        # Perform backward pass to calculate relevance contributions
        (z * s).sum().backward()
        c = a.grad
        # Calculate the propagated relevance for the input
        r = (a * c).data
        return r

class RelevancePropagationLinear(nn.Module):
    # Layer-wise relevance propagation for linear transformation using the z^+-rule
    def __init__(
        self,
        layer: torch.nn.Linear,
        eps: float = 1.0e-05,
        top_k: float = 0.0,
    ) -> None:
        super().__init__()

        self.layer = layer

        # Apply the z^+ rule: Keep only non-negative weights for relevance propagation
        self.layer.weight = torch.nn.Parameter(self.layer.weight.clamp(min=0.0))
        # Set biases to zero to focus on weighted input contributions
        self.layer.bias = torch.nn.Parameter(torch.zeros_like(self.layer.bias))

        self.eps = eps
        self.top_k = top_k

    @torch.no_grad()
    def forward(self, a: torch.tensor, r: torch.tensor) -> torch.tensor:
        # Filter the top-k% most relevant activations
        if self.top_k:
            r = relevance_filter(r, top_k_percent=self.top_k)
        # Forward pass to compute the activations plus stabilization term
        z = self.layer.forward(a) + self.eps
        # Compute the scaling factor for relevance distribution
        s = r / z
        # Backpropagate relevance through the weights
        c = torch.mm(s, self.layer.weight)
        # Compute input relevance by element-wise multiplication
        r = (a * c).data
        return r

class RelevancePropagationFlatten(nn.Module):
    # Layer-wise relevance propagation for flatten operation

    def __init__(self, layer: torch.nn.Flatten, top_k: float = 0.0) -> None:
        super().__init__()
        self.layer = layer

    @torch.no_grad()
    def forward(self, a: torch.tensor, r: torch.tensor) -> torch.tensor:
        # Reshape the relevance tensor to match the shape of the input tensor
        r = r.view(size=a.shape)
        return r

class RelevancePropagationReLU(nn.Module):
    # Layer-wise relevance propagation for ReLU activation
    # Passes the relevance scores without modification

    def __init__(self, layer: torch.nn.ReLU, top_k: float = 0.0) -> None:
        super().__init__()

    @torch.no_grad()
    def forward(self, a: torch.tensor, r: torch.tensor) -> torch.tensor:
        return r