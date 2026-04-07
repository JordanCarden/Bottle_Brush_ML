from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GINConv, global_add_pool

from .arch import Token, parse_architecture


class _GIN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int) -> None:
        super().__init__()

        def make_block(in_features: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(in_features, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )

        self.conv1 = GINConv(make_block(in_dim))
        self.conv2 = GINConv(make_block(hidden_dim))
        self.conv3 = GINConv(make_block(hidden_dim))
        self.lin1 = nn.Linear(hidden_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, 1)

    def forward(self, data: Data) -> torch.Tensor:  # type: ignore[override]
        x, edge_index, batch = data.x, data.edge_index, data.batch
        h = self.conv1(x, edge_index).relu()
        h = self.conv2(h, edge_index).relu()
        h = self.conv3(h, edge_index)
        h = global_add_pool(h, batch)
        h = self.lin1(h).relu()
        h = F.dropout(h, p=0.5, training=self.training)
        return self.lin2(h).view(-1)


def _tokens_to_graph(tokens: Sequence[Token]) -> Data:
    tokens_sorted = sorted(tokens, key=lambda t: t.position)
    features: List[List[int]] = []
    backbone_indices: dict[int, int] = {}
    for node_idx, tok in enumerate(tokens_sorted):
        features.append([1, 0, 0])
        backbone_indices[int(tok.position)] = node_idx

    edge_index: list[list[int]] = [[], []]
    for idx in range(len(tokens_sorted) - 1):
        edge_index[0].extend([idx, idx + 1])
        edge_index[1].extend([idx + 1, idx])

    next_node = len(features)
    for tok in tokens_sorted:
        if tok.length <= 0:
            continue
        prev = backbone_indices[int(tok.position)]
        for _ in range(int(tok.length)):
            features.append([0, 1, 0] if tok.chem == "S" else [0, 0, 1])
            curr = next_node
            edge_index[0].extend([prev, curr])
            edge_index[1].extend([curr, prev])
            prev = curr
            next_node += 1

    x = torch.tensor(features, dtype=torch.float32)
    edges = torch.tensor(edge_index, dtype=torch.long)
    return Data(x=x, edge_index=edges)


@dataclass(frozen=True, slots=True)
class GinPredictions:
    area: float
    rg: float
    rdf: float


class GinSurrogate:
    def __init__(
        self,
        *,
        repo_root: Path,
        device: str | None = None,
        hidden_dim: int = 128,
    ) -> None:
        self.repo_root = repo_root
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        models_dir = (repo_root / "RESULTS_RANDOM_ONLY" / "cv" / "GIN" / "models").resolve()
        with (models_dir / "normalization_stats.json").open("r", encoding="utf-8") as handle:
            stats = json.load(handle)

        self._stats = {
            "area": (float(stats["area"]["mean"]), float(stats["area"]["std"])),
            "rg": (float(stats["rg"]["mean"]), float(stats["rg"]["std"])),
            "rdf": (float(stats["rdf"]["mean"]), float(stats["rdf"]["std"])),
        }

        self._models: dict[str, _GIN] = {}
        for name in ("area", "rg", "rdf"):
            model = _GIN(in_dim=3, hidden_dim=hidden_dim).to(self.device)
            state = torch.load(models_dir / f"model_{name}.pt", map_location=self.device)
            model.load_state_dict(state)
            model.eval()
            self._models[name] = model

    def predict_many(
        self,
        architectures: Sequence[str],
        *,
        batch_size: int = 256,
    ) -> List[GinPredictions]:
        if not architectures:
            return []

        graphs: List[Data] = []
        for arch in architectures:
            tokens = parse_architecture(arch)
            graphs.append(_tokens_to_graph(tokens))

        out: List[GinPredictions] = []
        with torch.no_grad():
            for start in range(0, len(graphs), batch_size):
                batch_graph = Batch.from_data_list(graphs[start : start + batch_size]).to(self.device)
                preds: dict[str, torch.Tensor] = {}
                for name, model in self._models.items():
                    preds[name] = model(batch_graph).view(-1).cpu()
                for idx in range(batch_graph.num_graphs):
                    area = float(preds["area"][idx].item() * self._stats["area"][1] + self._stats["area"][0])
                    rg = float(preds["rg"][idx].item() * self._stats["rg"][1] + self._stats["rg"][0])
                    rdf = float(preds["rdf"][idx].item() * self._stats["rdf"][1] + self._stats["rdf"][0])
                    out.append(GinPredictions(area=area, rg=rg, rdf=rdf))
        return out
