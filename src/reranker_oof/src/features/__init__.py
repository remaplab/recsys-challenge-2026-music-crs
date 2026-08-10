"""Feature-engineering layer for the reranker.

Modules
-------
- ``pool``            : per-CG long-format conversion, outer-join pool, fusion features
- ``resources``       : data loaders (track metadata, user metadata, URM, history)
- ``feature_builder`` : tabular FeatureBuilder (families B/C/E/F/G/H/I)
- ``cg_calibration``  : per-CG probability calibration + conformal set-size
                        (leave-one-fold-out fit; emits
                        ``calibrated_score_<cg>`` + ``set_size_<cg>`` cols)
- ``holdout_subsets`` : subset generators for stratified holdout evaluation
"""
