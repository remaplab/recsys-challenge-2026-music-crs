import pickle
from abc import ABC, abstractmethod
from pathlib import Path
import polars as pl


class BaseRecommender(ABC):
    """
    Base class for session-aware candidate generators.

    Contract:
        fit(train_df)   — learn from training sessions
        recommend(...)  — return top-k candidates per (session, turn)

    Output schema of recommend():
        session_id : str/int   — session identifier
        turn       : int       — turn index within session
        track_ids  : list[str] — ranked candidate track UUIDs (len <= top_k)
        scores     : list[float] | None — per-candidate scores, same order as track_ids

    Persistence:
        Override _get_model_state() / _set_model_state() in subclasses to
        include fitted attributes. Base save/load handles the rest.
    """

    RECOMMENDER_NAME = "BaseRecommender"

    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def fit(self, train_df: pl.DataFrame, **kwargs) -> None:
        """Fit the model on training sessions."""
        ...

    @abstractmethod
    def recommend(
        self,
        test_df: pl.DataFrame,
        top_k: int = 20,
        remove_seen: bool = True,
        **kwargs,
    ) -> pl.DataFrame:
        """
        Generate top-k candidates for every (session, turn) in test_df.
        From most relevant to least relevant.

        Returns a DataFrame with columns:
            session_id, turn, track_ids (list), scores (list or null)
        """
        ...

    def save(self, path: str | Path) -> None:
        """Pickle model state to path (parent dirs created automatically)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {"recommender_name": self.RECOMMENDER_NAME}
        state.update(self._get_model_state())
        with open(path, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"    ✅ {self.RECOMMENDER_NAME} saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "BaseRecommender":
        """Load a previously saved model. Does not require re-passing training data."""
        with open(path, "rb") as f:
            state = pickle.load(f)
        instance = cls.__new__(cls)
        instance._set_model_state(state)
        print(f"    ✅ {state['recommender_name']} loaded from {path}")
        return instance

    def _get_model_state(self) -> dict:
        """Return fitted attributes to persist. Override in subclasses."""
        return {}

    def _set_model_state(self, state: dict) -> None:
        """Restore fitted attributes from loaded state. Override in subclasses."""
        pass
