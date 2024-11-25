from copy import deepcopy

import torch
from torch import nn

from utils import layers_lookup


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
        """Method builds the model for layer-wise relevance propagation.

        Returns:
            LRP-model as module list.

        """
        # Clone layers from original model. This is necessary as we might modify the weights.
        layers = deepcopy(self.layers)
        lookup_table = layers_lookup()

        # Run backwards through layers
        for i, layer in enumerate(layers[::-1]):
            try:
                layers[i] = lookup_table[layer.__class__](layer=layer, top_k=self.top_k)
            except KeyError:
                message = (
                    f"Layer-wise relevance propagation not implemented for "
                    f"{layer.__class__.__name__} layer."
                )
                raise NotImplementedError(message)

        return layers




    def _get_layer_operations(self) -> torch.nn.ModuleList:
        '''
        layers = torch.nn.ModuleList()

        # Manually add each layer in the order of forward pass for Net
        layers.append(self.model.conv1)
        #      layers.append(self.model.pool)
        layers.append(self.model.conv2)
        #      layers.append(self.model.pool)

        # Flatten layer (replaces torch.flatten in forward)
        layers.append(torch.nn.Flatten(start_dim=1))

        # Add fully connected layers
        layers.append(self.model.fc1)
        layers.append(self.model.fc2)
        layers.append(self.model.fc3)

        return layers
        '''

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