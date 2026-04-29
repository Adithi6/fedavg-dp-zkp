import logging
import time
from torch.utils.data import DataLoader
import numpy as np

from client.fl_client import FederatedClient
from utils.weights import bytes_to_weight_arrays, apply_weight_arrays


class GossipNode:
    """
    Gossip node for Model 3:
    - FedAvg
    - Differential Privacy
    - Gossip communication
    - Schnorr ZKP verification is handled in gossip/protocol.py
    - Update validation is handled before aggregation
    """

    def __init__(
        self,
        client_id: str,
        dataloader: DataLoader,
        device: str,
        weight_dtype: str,
        learning_rate: float,
        model_name: str,
        input_channels: int,
        num_classes: int,
        input_height: int,
        input_width: int,
        conv1_channels: int,
        conv2_channels: int,
        hidden_dim: int,
    ):
        self.client = FederatedClient(
            client_id=client_id,
            dataloader=dataloader,
            device=device,
            weight_dtype=weight_dtype,
            learning_rate=learning_rate,
            model_name=model_name,
            input_channels=input_channels,
            num_classes=num_classes,
            input_height=input_height,
            input_width=input_width,
            conv1_channels=conv1_channels,
            conv2_channels=conv2_channels,
            hidden_dim=hidden_dim,
        )

        self.client_id = client_id
        self.own_submission: dict | None = None
        self.inbox: dict[str, dict] = {}

        logging.info(
            f"[{self.client_id}] gossip node initialized | "
            f"weight_dtype={weight_dtype} learning_rate={learning_rate} "
            f"model={model_name}"
        )

    def local_train(self, global_weight_arrays: list | None, epochs: int = 1):
        self.client.local_train(global_weight_arrays, epochs)

    def prepare_update(self) -> dict:
        """
        Creates a model update with Schnorr ZKP proof.
        """
        self.own_submission = self.client.prepare_update()
        self.inbox.clear()

        logging.info(f"[{self.client_id}] ZKP update prepared and inbox reset")
        return self.own_submission

    def receive_gossip(self, message: dict):
        origin_id = message["client_id"]

        if origin_id == self.client_id:
            logging.warning(
                f"[{self.client_id}] ignored returned own gossip from {origin_id}"
            )
            return

        if origin_id in self.inbox:
            logging.warning(
                f"[{self.client_id}] duplicate gossip ignored from {origin_id}"
            )
            return

        self.inbox[origin_id] = message

        logging.info(
            f"[{self.client_id}] received gossip from {origin_id} "
            f"| inbox_size={len(self.inbox)}"
        )

    def get_all_submissions(self) -> list[dict]:
        all_subs = []

        if self.own_submission is not None:
            all_subs.append(self.own_submission)

        all_subs.extend(self.inbox.values())
        return all_subs

    def clear_submissions(self):
        self.own_submission = None
        self.inbox.clear()
        logging.info(f"[{self.client_id}] cleared round submissions")

    def is_valid_update(self, arrays: list, max_norm: float = 100.0) -> bool:
        """
        Lightweight abnormal-update validation.
        Rejects:
        - NaN values
        - Infinite values
        - Extremely large model updates

        This does not replace full zk-SNARK training correctness proof,
        but helps reject abnormal/malicious updates.
        """
        total_norm = 0.0

        for arr in arrays:
            if np.isnan(arr).any() or np.isinf(arr).any():
                return False

            total_norm += np.linalg.norm(arr)

        return total_norm <= max_norm

    def aggregate_local_updates(self, submissions: list[dict], template_model):
        start_time = time.time()
        
        if not submissions:
            logging.warning(f"[{self.client_id}] no submissions available for aggregation")
            return

        logging.info(f"[{self.client_id}] received {len(submissions)} submission(s) for aggregation")

        dtype_name = self.client.weight_dtype

        weight_sets = []
        accepted_clients = []
        rejected_clients = []

        for sub in submissions:
            arrays = bytes_to_weight_arrays(
                sub["update_bytes"],
                template_model,
                dtype_name=dtype_name,
            )

            if not self.is_valid_update(arrays):
                rejected_clients.append(sub["client_id"])
                logging.warning(
                    f"[{self.client_id}] rejected abnormal update from {sub['client_id']}"
                )
                continue

            weight_sets.append(arrays)
            accepted_clients.append(sub["client_id"])

        if not weight_sets:
            logging.warning(f"[{self.client_id}] no valid updates after validation")
            return

        logging.info(
            f"[{self.client_id}] valid updates accepted: "
            f"{len(accepted_clients)}/{len(submissions)} | clients={accepted_clients}"
        )

        if rejected_clients:
            logging.warning(
                f"[{self.client_id}] rejected clients: {rejected_clients}"
            )

        averaged = [
            np.mean([weights[i] for weights in weight_sets], axis=0)
            for i in range(len(weight_sets[0]))
        ]

        apply_weight_arrays(self.client.model, averaged)

        end_time = time.time()
        logging.info(f"[{self.client_id}] validated FedAvg aggregation completed")
        logging.info(f"[{self.client_id}] aggregate_local_updates overall execution time: {end_time - start_time:.4f} seconds")