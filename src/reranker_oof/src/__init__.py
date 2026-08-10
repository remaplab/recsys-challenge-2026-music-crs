"""Library code for the OOF reranker pipeline.

Submodules
----------
- ``paths``   : single source of truth for repo/data/model paths
- ``eval``    : nDCG / recall macro-by-turn (the project's official metric)
- ``features``: pool-union, fusion features, FeatureBuilder, holdout subsets
- ``rerankers``: abstract base + XGBoost / LightGBM / CatBoost / NN / SVGD-NN

The launchers in this package are thin CLI wrappers around this code.
"""
