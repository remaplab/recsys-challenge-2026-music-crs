import pickle
from abc import ABC, abstractmethod
from pathlib import Path
import polars as pl

class BaseRecommender(ABC):

    RECOMMENDER_NAME = "BaseRecommender"

    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def fit(self, train_df: pl.DataFrame, **kwargs) -> None:
        ...

    @abstractmethod
    def recommend(
        self,
        test_df: pl.DataFrame,
        top_k: int = 20,
        remove_seen: bool = True,
        **kwargs,
    ) -> pl.DataFrame:
        ...

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {"recommender_name": self.RECOMMENDER_NAME}
        state.update(self._get_model_state())
        with open(path, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"    ✅ {self.RECOMMENDER_NAME} saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "BaseRecommender":
        with open(path, "rb") as f:
            state = pickle.load(f)
        instance = cls.__new__(cls)
        instance._set_model_state(state)
        print(f"    ✅ {state['recommender_name']} loaded from {path}")
        return instance

    def _get_model_state(self) -> dict:
        return {}

    def _set_model_state(self, state: dict) -> None:
        pass
