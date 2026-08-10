"""SessionRecommender — base class for sequential / session-aware recommenders.

Thin wrapper over UserRecommender that fixes _filter_candidate_mask to use
self.max_future_years instead of requiring it as a call-site argument.
Sequential models (bert4rec, feature_bert4rec) inherit from this class.
"""

from __future__ import annotations

from datetime import date

import numpy as np

from .user_base import UserRecommender


class SessionRecommender(UserRecommender):
    """Base for session-sequential recommenders.

    Identical to UserRecommender except _filter_candidate_mask uses
    self.max_future_years by default, matching the one-arg call convention
    used in bert4rec / feature_bert4rec recommend() loops.
    """

    RECOMMENDER_NAME = "SessionRecommender"

    def _filter_candidate_mask(  # type: ignore[override]
        self, session_date: date | None
    ) -> np.ndarray | None:
        return super()._filter_candidate_mask(session_date, self.max_future_years)
