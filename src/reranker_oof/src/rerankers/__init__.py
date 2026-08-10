"""Reranker implementations behind a single ``BaseReranker`` ABC.

Implemented backends
--------------------
- ``xgb`` : XGBoost ``rank:ndcg``, GPU histogram, streaming ``QuantileDMatrix``
"""
