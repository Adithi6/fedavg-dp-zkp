# Secure Decentralized Federated Learning using Gossip Protocol and Zero-Knowledge Proofs

## Overview

This project implements a secure decentralized Federated Learning framework that combines Differential Privacy (DP), Gossip-based communication, and Zero-Knowledge Proof (ZKP) based model update validation.

Unlike traditional Federated Learning systems that depend on a central server, clients communicate directly with neighboring peers using a gossip protocol. Before model updates are accepted for aggregation, they are validated using zk-SNARK proofs generated through Circom and snarkjs.

The framework aims to provide privacy preservation, decentralized communication, and protection against malicious model updates while maintaining learning performance.

---

## Key Features

* Decentralized Federated Learning (No Central Server)
* Gossip-Based Peer-to-Peer Communication
* Differential Privacy Protection
* Zero-Knowledge Proof Validation
* SHA-256 Update Hashing
* Non-IID Data Distribution
* LeNet-based CNN Model
* Circom zk-SNARK Integration
* Model Update Verification Before Aggregation

---

## Security Mechanisms

### Differential Privacy

Protects client updates through:

* Gradient clipping
* Noise injection
* Privacy budget control (ε, δ)

### SHA-256 Hashing

Each model update is hashed before proof generation.

Benefits:

* Integrity protection
* Fixed-size representation of updates
* Input binding for proof generation

### Zero-Knowledge Proofs

The framework uses zk-SNARKs to prove that a model update satisfies predefined constraints without revealing the update itself.

Benefits:

* Update validation
* Privacy-preserving verification
* Protection against malformed or malicious updates
* Compact proof generation and verification

### Gossip Protocol

Provides:

* Fully decentralized communication
* Fault tolerance
* Scalability
* Server-free update propagation

---

## System Workflow

1. Initialize federated clients.
2. Distribute MNIST data using non-IID partitioning.
3. Train local models.
4. Apply Differential Privacy.
5. Compute SHA-256 hash of model updates.
6. Generate zk-SNARK proof.
7. Gossip model updates to neighboring peers.
8. Verify received proofs.
9. Accept only valid updates.
10. Aggregate verified updates using FedAvg.
11. Repeat for multiple communication rounds.

---

## Experimental Setup

| Parameter            | Value                       |
| -------------------- | --------------------------- |
| Dataset              | MNIST                       |
| Model                | LeNet-style CNN             |
| Clients              | 10                          |
| Communication Rounds | 150                         |
| Local Epochs         | 5                           |
| Data Distribution    | Non-IID Dirichlet (α = 0.5) |
| Communication        | Gossip Protocol             |
| Fanout               | 2                           |
| Max Hops             | 3                           |
| Optimizer            | Adam                        |
| Learning Rate        | 0.001                       |
| Batch Size           | 64                          |
| DP Epsilon (ε)       | 1.0                         |
| DP Delta (δ)         | 1e-5                        |
| DP Clip Norm         | 0.5                         |
| ZKP Sample Size      | 10                          |
| Scale Factor         | 1000                        |
| Validation Threshold | 100000000                   |
| Platform             | Google Colab                |

---

## Repository Structure

```text
fedavg_dp_zkp/
│
├── client/
├── crypto/
├── data/
├── gossip/
├── model/
├── utils/
├── zkp/
│
├── config.yaml
├── main.py
├── fedavg_dp_zkp.ipynb
└── README.md
```

---

## Installation

```bash
pip install torch torchvision
pip install datasets
pip install flwr
pip install flwr-datasets
pip install numpy pandas pyyaml
```

For zk-SNARK generation:

```bash
npm install -g snarkjs
```

Install Circom separately if local proof generation is required.

---

## Running on Google Colab

```python
from google.colab import files
files.upload()

!unzip fedavg_dp_zkp.zip

%cd fedavg_dp_zkp

!pip install torch torchvision datasets pyyaml numpy pandas flwr flwr-datasets

!python main.py
```

---

## Configuration

Key parameters are defined in `config.yaml`.

```yaml
dp:
  enabled: true
  clip_norm: 0.5
  target_epsilon: 1.0
  delta: 1e-5
  auto_noise: true
  base_noise: 0.05

zkp:
  enabled: true
  proof_system: circom_snark
  verify_before_aggregation: true
  sample_size: 10
  scale: 1000
  threshold: 100000000

gossip:
  fanout: 2
  max_hops: 3
```

---

## Research Motivation

Federated Learning protects raw training data by keeping it on client devices. However, malicious participants can still submit incorrect or poisoned model updates.

This work investigates whether Zero-Knowledge Proofs can be used to validate model updates before aggregation while preserving privacy. The framework combines:

* Differential Privacy for privacy preservation
* Gossip Protocol for decentralized communication
* Zero-Knowledge Proofs for update validation

The goal is to study the trade-off between security, privacy, computational overhead, and model performance in decentralized Federated Learning environments.

---

## Author

Adithi

B.Tech, Computer Science and Engineering

NMAM Institute of Technology (NMAMIT)

Research Intern, National Institute of Technology Karnataka (NITK), Surathkal
