from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _import_lr(repo_root: Path):
    lr_dir = (repo_root / "LR").resolve()
    if str(lr_dir) not in sys.path:
        sys.path.insert(0, str(lr_dir))
    import preprocess as lr_preprocess  # type: ignore
    import train as lr_train  # type: ignore

    return lr_preprocess, lr_train


@dataclass(frozen=True, slots=True)
class OODResult:
    score: float
    percentile: float
    flag: int


class OODScorer:
    def __init__(
        self,
        *,
        repo_root: Path,
        random_csv: Path | None = None,
        pca_components: int = 10,
        percentile_threshold: float = 99.0,
        drop_redundant: bool = True,
    ) -> None:
        self.repo_root = repo_root
        self.random_csv = random_csv or (repo_root / "data" / "ES.csv")
        self.percentile_threshold = float(percentile_threshold)

        lr_preprocess, lr_train = _import_lr(repo_root)
        self._lr_preprocess = lr_preprocess

        df_random = lr_preprocess.create_feature_dataframe(str(self.random_csv))
        df_random = df_random.rename(columns=lambda c: str(c).strip())

        targets = {"Area AVG", "RG AVG", "RDF Peak"}
        feature_cols = [
            col
            for col in df_random.columns
            if col not in targets
            and col != "Input List"
            and pd.api.types.is_numeric_dtype(df_random[col])
        ]
        if drop_redundant:
            for col in ("max_length", "min_block_size"):
                if col in feature_cols:
                    feature_cols.remove(col)
        self.feature_cols = feature_cols

        pipeline_steps = [
            ("prep", lr_train.build_preprocessor(feature_cols)),
            ("scale", StandardScaler()),
        ]
        if pca_components and pca_components > 0:
            pipeline_steps.append(
                ("pca", PCA(n_components=int(pca_components), random_state=0, whiten=True))
            )
        self.pipeline = Pipeline(pipeline_steps)

        X_random = df_random.loc[:, feature_cols]
        Z_random = self.pipeline.fit_transform(X_random)
        scores = np.linalg.norm(Z_random, axis=1)
        self._train_scores_sorted = np.sort(scores)
        self.threshold = float(np.percentile(scores, self.percentile_threshold))

    def score_many(self, architectures: Sequence[str]) -> List[OODResult]:
        if not architectures:
            return []

        df = pd.DataFrame({"_arch": list(architectures)})
        df["Input List"] = df["_arch"].apply(self._lr_preprocess.parse_vectors)
        empty = df["Input List"].apply(len) == 0
        if bool(empty.any()):
            bad_examples = df.loc[empty, "_arch"].head(3).tolist()
            raise ValueError(f"Failed to parse some architectures for OOD scoring: {bad_examples!r}")
        df = self._lr_preprocess.add_features(df)
        X = df.loc[:, self.feature_cols]
        Z = self.pipeline.transform(X)
        scores = np.linalg.norm(Z, axis=1)

        n = len(self._train_scores_sorted)
        percentiles = np.searchsorted(self._train_scores_sorted, scores, side="right") / n * 100.0
        return [
            OODResult(
                score=float(score),
                percentile=float(pct),
                flag=int(score > self.threshold),
            )
            for score, pct in zip(scores, percentiles, strict=True)
        ]
