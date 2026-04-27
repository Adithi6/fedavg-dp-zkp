import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model.cnn import SmallCNN
from crypto import zkp_utils
from utils.weights import apply_weight_arrays, weights_to_bytes


def build_model(
    model_name,
    device,
    input_channels,
    num_classes,
    input_height,
    input_width,
    conv1_channels,
    conv2_channels,
    hidden_dim,
):
    if model_name.lower() == "smallcnn":
        return SmallCNN(
            input_channels=input_channels,
            num_classes=num_classes,
            input_height=input_height,
            input_width=input_width,
            conv1_channels=conv1_channels,
            conv2_channels=conv2_channels,
            hidden_dim=hidden_dim,
        ).to(device)

    raise ValueError("Unsupported model")


class FederatedClient:
    def __init__(
        self,
        client_id,
        dataloader,
        device,
        weight_dtype,
        learning_rate,
        model_name,
        input_channels,
        num_classes,
        input_height,
        input_width,
        conv1_channels,
        conv2_channels,
        hidden_dim,
    ):
        self.client_id = client_id
        self.dataloader = dataloader
        self.device = device
        self.weight_dtype = weight_dtype
        self.learning_rate = learning_rate

        # DP params
        self.dp_clip_norm = 1.0
        self.dp_noise_std = 0.01

        # ZKP keys
        self.pk, self.sk, _ = zkp_utils.keygen()

        self.model = build_model(
            model_name,
            device,
            input_channels,
            num_classes,
            input_height,
            input_width,
            conv1_channels,
            conv2_channels,
            hidden_dim,
        )

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def local_train(self, global_weight_arrays=None, epochs=1):
        if global_weight_arrays is not None:
            apply_weight_arrays(self.model, global_weight_arrays)

        self.model.train()

        for _ in range(epochs):
            for x, y in self.dataloader:
                x, y = x.to(self.device), y.to(self.device)

                self.optimizer.zero_grad()
                loss = self.criterion(self.model(x), y)
                loss.backward()

                # DP clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.dp_clip_norm
                )

                # DP noise
                for param in self.model.parameters():
                    if param.grad is not None:
                        param.grad += torch.normal(
                            0,
                            self.dp_noise_std,
                            size=param.grad.shape,
                            device=param.grad.device,
                        )

                self.optimizer.step()

    def prepare_update(self):
        update_bytes = weights_to_bytes(self.model, self.weight_dtype)

        # Schnorr signature (identity proof)
        schnorr_proof = zkp_utils.generate_proof(
            self.sk,
            update_bytes,
            self.client_id,
        )

        # Real zk-SNARK (model norm proof)
        # We sample 10 weights from the model to prove (since our circuit size is 10)
        from utils.weights import model_to_weight_arrays
        import numpy as np
        arrays = model_to_weight_arrays(self.model)
        flat = np.concatenate([arr.flatten() for arr in arrays])
        # scale and make positive for circom
        sampled_weights = np.abs(flat[:10] * 1000).astype(int).tolist()
        
        snark_proof, snark_public, _ = zkp_utils.generate_snark(sampled_weights, threshold=100000000)

        return {
            "client_id": self.client_id,
            "update_bytes": update_bytes,
            "zkp": schnorr_proof,
            "snark_proof": snark_proof,
            "snark_public": snark_public,
            "public_key": self.pk,
        }