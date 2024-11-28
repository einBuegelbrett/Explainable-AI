import torch
from torch import nn
from copy import deepcopy

from lrp.lrp_layers import RelevancePropagationConv2d, RelevancePropagationLinear, RelevancePropagationReLU, RelevancePropagationFlatten

class LRPModel(nn.Module):
    def __init__(self, model: torch.nn.Module, top_k: float = 0.0) -> None:
        super().__init__()
        self.model = model
        self.top_k = top_k
        self.model.eval()  # self.model.train() activates dropout / batch normalization etc.!

        # Parse network
        self.layers = self._get_layer_operations()

        # Create LRP network
        self.lrp_layers = self._create_lrp_model()

    def _create_lrp_model(self) -> torch.nn.ModuleList:
        '''
        This method builds the model for layer-wise relevance propagation adapted to our net.
        '''

        # Clone layers from original model. This is necessary as we might modify the weights.
        # Manually map each layer to its corresponding LRP operation
        lrp_layers = nn.ModuleList()

        # Reverse the layers because LRP works backwards through the network
        for layer in reversed(self.layers):
            # Deep copy the layer to avoid modifying the original
            layer_copy = deepcopy(layer)

            if isinstance(layer, nn.Conv2d):
                lrp_layer = RelevancePropagationConv2d(layer=layer, top_k=self.top_k)
            elif isinstance(layer, nn.Linear):
                lrp_layer = RelevancePropagationLinear(layer=layer, top_k=self.top_k)
            elif isinstance(layer, nn.ReLU):
                lrp_layer = RelevancePropagationReLU(layer=layer, top_k=self.top_k)
            elif isinstance(layer, nn.Flatten):
                lrp_layer = RelevancePropagationFlatten(layer=layer, top_k=self.top_k)
            else:
                raise NotImplementedError(
                    f"Layer-wise relevance propagation not implemented for {layer.__class__.__name__} layer.")
            lrp_layers.append(lrp_layer)

        return lrp_layers

    def _get_layer_operations(self) -> torch.nn.ModuleList:
        layers = torch.nn.ModuleList()
        layers.append(self.model.conv1)
        layers.append(self.model.relu1)
        layers.append(self.model.conv2)
        layers.append(self.model.relu2)
        layers.append(self.model.flatten)
        layers.append(self.model.fc1)
        layers.append(self.model.relu3)
        layers.append(self.model.fc2)
        layers.append(self.model.fc3)
        
        return layers 

    def forward(self, x: torch.tensor) -> torch.tensor:
        activations = list()

        # Run inference and collect activations.
        with torch.no_grad():
            # Replace image with ones avoids using image information for relevance computation.
            activations.append(torch.ones_like(x))
            for layer in self.layers:
                x = layer.forward(x)
                activations.append(x)

        # Reverse order of activations to run backwards through model
        activations = activations[::-1]
        activations = [a.data.requires_grad_(True) for a in activations]

        # Initial relevance scores are the network's output activations
        relevance = torch.softmax(activations.pop(0), dim=-1)  # Unsupervised

        # Perform relevance propagation
        for i, layer in enumerate(self.lrp_layers):
            relevance = layer.forward(activations.pop(0), relevance)

        return relevance.permute(0, 2, 3, 1).sum(dim=-1).squeeze().detach().cpu()