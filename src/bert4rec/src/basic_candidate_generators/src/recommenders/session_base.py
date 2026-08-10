from __future__ import annotations

from datetime import date

import numpy as np

from .user_base import UserRecommender

class SessionRecommender(UserRecommender):

    RECOMMENDER_NAME = "SessionRecommender"

    def _filter_candidate_mask(
        self, session_date: date | None
    ) -> np.ndarray | None:
        return super()._filter_candidate_mask(session_date, self.max_future_years)
