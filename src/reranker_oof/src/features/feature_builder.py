from __future__ import annotations

import time
from functools import cached_property
from pathlib import Path
from typing import Iterable

import polars as pl


_DEFAULT_TEST_TRACKS_PATH = (
    Path(__file__).resolve().parents[4]
    / "data/talkpl-ai/TalkPlayData-Challenge-Track-Metadata/data/test_tracks-00000-of-00001.parquet"
)


BEHAVIORAL_CGS = {
    "bm25", "tfidf",
    "bm25_cg_session", "tfidf_cg_session",
    "bm25_cg_session_dro", "tfidf_cg_session_dro",
    "item_knn_session", "item_knn_user",
    "rp3beta_session", "rp3beta_user",
    "slim_bpr_session", "slim_bpr_user",
    "slim_elasticnet_session", "slim_elasticnet_user",
    "sequential_rules_session",
    "heuristic_session", "heuristic_session_dro",
}
SEMANTIC_CGS = {
    "recvae_session", "recvae_user",
    "multvae_session", "multvae_user",
    "ials_session", "ials_user",
    "pure_svd_session", "pure_svd_user",
    "lightfm_icm_session",
    "prod2vec_session", "prod2vec_user",
    "gf_cf_session",
    "emb_item_knn_8b_session_dro",
    "two_tower_v2_session",
    "split_hidim_xattn_hardneg_glu_session",
    "split_hidim_xattn_hardneg_moddrop_session",
    "split_hidim_xattn_hardneg_query_session",
    "split_hidim_xattn_hardneg_query_full_session",
    "split_hidim_xattn_hardneg_query_full_trainfusion_session",
    "split_hidim_xattn_hardneg_query_full_mha_userhistmod_sess_v2_session",
    "hybrid_8b_session_dro", "hybrid_all_qwen_session_dro",
    "tower_ensemble_session_dro",
}


DEFAULT_MOOD_LEXICON: tuple[str, ...] = (
    "happy", "sad", "energetic", "chill", "relaxing", "upbeat",
    "melancholy", "romantic", "angry", "calm", "party", "workout",
)


CATEGORY_LEVELS: tuple[str, ...] = tuple("ABCDEFGHIJK")
SPECIFICITY_LEVELS: tuple[str, ...] = ("HH", "LH", "HL", "LL")


class FeatureBuilder:

    STAGES: tuple[tuple[str, str], ...] = (
        ("F+I (session+turn)",  "_add_F_I"),
        ("C  (track stats)",    "_add_C"),
        ("K  (goal+text)",      "_add_K_text_goal"),
        ("J  (t-1 dynamics)",   "_add_J_dynamics"),
        ("E  (overlap)",        "_add_E"),
        ("G  (user meta)",      "_add_G"),
        ("B  (CG families)",    "_add_B"),
        ("H  (text only)",      "_add_H_text_only"),
        ("M  (embeddings)",     "_add_M_embeddings"),
        ("X  (cross-features)", "_add_X_crossfeatures"),
        ("BC (blind-catalog)",  "_add_BC_blind_catalog"),
    )

    def __init__(
        self,
        track_meta: pl.DataFrame,
        user_meta: pl.DataFrame,
        warm_user_ids: Iterable[str],
        urm_df: pl.DataFrame,
        session_history_df: pl.DataFrame,
        mood_lexicon: tuple[str, ...] = DEFAULT_MOOD_LEXICON,
        emb_resources=None,
        test_tracks_path: "str | Path | None" = _DEFAULT_TEST_TRACKS_PATH,
        emb_cache_path: "str | Path | None" = None,
    ):
        self.track_meta = track_meta
        self.user_meta = user_meta
        
        self.test_tracks_path = test_tracks_path
        
        self.warm_user_ids = set(warm_user_ids)
        self.urm_df = urm_df
        self.session_history_df = session_history_df
        self.mood_lexicon = mood_lexicon
        
        self.emb_resources = emb_resources
        
        self.emb_cache_path = emb_cache_path


    @cached_property
    def _track_stats(self) -> pl.DataFrame:
        
        pop = self.urm_df.group_by("track_id").agg(pl.len().alias("track_play_count"))

        m = self.track_meta.with_columns(
            pl.col("artist_id").list.first().alias("artist_id_first"),
            pl.col("album_id").list.first().alias("album_id_first"),
            pl.col("tag_list").list.len().alias("n_tags"),
            pl.col("artist_name").list.first().alias("artist_name_first"),
            pl.col("album_name").list.first().alias("album_name_first"),
            pl.col("track_name").list.first().alias("track_name_first"),
            
            pl.col("duration").alias("track_duration"),
            pl.col("popularity").alias("track_meta_popularity"),
            pl.col("release_date").str.slice(0, 4)
              .cast(pl.Int32, strict=False).alias("track_release_year"),
        ).join(pop, on="track_id", how="left").with_columns(
            pl.col("track_play_count").fill_null(0)
        ).with_columns(
            
            (pl.col("track_release_year").is_null()
             | (pl.col("track_duration").fill_null(0) <= 0)).alias("is_blind_catalog"),
        ).with_columns(
            
            (pl.col("track_release_year").is_null().cast(pl.Int8)
             + (pl.col("track_duration").fill_null(0) <= 0).cast(pl.Int8)
             + (pl.col("track_meta_popularity").fill_null(0.0) <= 0).cast(pl.Int8)
             + (pl.col("n_tags") == 0).cast(pl.Int8)
             + pl.col("album_name_first").is_null().cast(pl.Int8)
            ).alias("n_missing_meta_fields"),
        )

        
        pos = m.filter(pl.col("track_meta_popularity") > 0)["track_meta_popularity"]
        mu = float(pos.mean()) if pos.len() else 0.0
        sigma = float(pos.std()) if pos.len() and pos.std() else 1.0
        q33 = float(pos.quantile(0.33)) if pos.len() else 0.0
        q67 = float(pos.quantile(0.67)) if pos.len() else 0.0
        m = m.with_columns(
            ((pl.col("track_meta_popularity").fill_null(0.0) - mu) / sigma)
                .alias("meta_pop_z"),
            pl.when(pl.col("track_meta_popularity").fill_null(0.0) <= 0)
              .then(pl.lit(-1, dtype=pl.Int8))
              .when(pl.col("track_meta_popularity") < q33)
              .then(pl.lit(0, dtype=pl.Int8))
              .when(pl.col("track_meta_popularity") < q67)
              .then(pl.lit(1, dtype=pl.Int8))
              .otherwise(pl.lit(2, dtype=pl.Int8))
              .alias("track_pop_bucket"),
        )

        
        m = m.with_columns(
            pl.col("track_play_count").rank(method="average").alias("_pr")
        ).with_columns(
            (pl.col("_pr") / pl.col("_pr").max()).alias("track_pop_pct")
        ).drop("_pr")

        
        art_pop = m.group_by("artist_id_first").agg(
            pl.col("track_play_count").sum().alias("artist_play_count")
        )
        alb_pop = m.group_by("album_id_first").agg(
            pl.col("track_play_count").sum().alias("album_play_count")
        )
        alb_card = m.group_by("album_id_first").agg(
            pl.len().alias("album_track_count")
        )

        m = (
            m.join(art_pop,  on="artist_id_first", how="left")
             .join(alb_pop,  on="album_id_first",  how="left")
             .join(alb_card, on="album_id_first",  how="left")
             .with_columns(
                 (pl.col("track_pop_pct") >= 0.99).alias("is_top1pct"),
                 (pl.col("track_pop_pct") <= 0.50).alias("is_long_tail"),
             )
        )
        return m.select(
            "track_id", "artist_id_first", "album_id_first",
            "artist_name_first", "album_name_first", "track_name_first",
            "track_play_count", "track_pop_pct",
            "artist_play_count", "album_play_count",
            "n_tags", "album_track_count",
            "is_top1pct", "is_long_tail",
            "track_duration", "track_meta_popularity", "track_release_year",
            "meta_pop_z", "track_pop_bucket",
            "is_blind_catalog", "n_missing_meta_fields",
            "tag_list",
        )

    @cached_property
    def _user_meta_clean(self) -> pl.DataFrame:
        """User metadata subset used by family G (one row per user)."""
        return self.user_meta.select(
            "user_id", "age", "age_group", "country_code", "gender"
        )

    @cached_property
    def _user_culture(self) -> pl.DataFrame | None:
        """Disabled — the culture-overlap Jaccard feature was dropped."""
        return None

    @cached_property
    def _session_ctx(self) -> tuple[pl.DataFrame, pl.DataFrame]:
    
        hist = self.session_history_df.select(
            "session_id",
            pl.col("turn_number").alias("hist_turn"),
            "track_id",
        )
        target_turns = hist.select(
            "session_id", pl.col("hist_turn").alias("turn_number")
        ).unique()

        ctx = (
            target_turns.join(hist, on="session_id")
            .filter(pl.col("hist_turn") < pl.col("turn_number"))
            .group_by("session_id", "turn_number")
            .agg(pl.col("track_id").alias("ctx_track_ids"))
        )
        all_targets = self.session_history_df.group_by("session_id").agg(
            pl.col("turn_number").max().alias("max_turn")
        )
        return ctx, all_targets
    
    @cached_property
    def _last_turn_ctx(self) -> pl.DataFrame:
        """One row per (session, turn) containing the track_id and stats of turn t-1."""
        hist = self.session_history_df.select(
            "session_id",
            (pl.col("turn_number") + 1).alias("turn_number"),  
            pl.col("track_id").alias("last_track_id")
        )
        return hist.join(
            self._track_stats.select(
                pl.col("track_id").alias("last_track_id"),
                pl.col("artist_id_first").alias("last_artist_id"),
                pl.col("album_id_first").alias("last_album_id"),
                pl.col("track_play_count").alias("last_pop"),
                pl.col("track_release_year").alias("last_year"),
                pl.col("track_duration").alias("last_duration"),
                pl.col("track_pop_bucket").alias("last_pop_bucket"),
                pl.col("tag_list").alias("last_tags").fill_null([])
            ),
            on="last_track_id", how="inner"
        )

    @cached_property
    def _ctx_long(self) -> pl.DataFrame:
        
        ctx, _ = self._session_ctx
        return (
            ctx.explode("ctx_track_ids").rename({"ctx_track_ids": "track_id"})
            .join(
                self._track_stats.select(
                    "track_id",
                    pl.col("artist_id_first").alias("_past_artist"),
                    pl.col("album_id_first").alias("_past_album"),
                    pl.col("track_play_count").alias("_past_pop"),
                    pl.col("track_duration").alias("_past_duration"),
                    pl.col("track_release_year").alias("_past_year"),
                    pl.col("is_blind_catalog").alias("_past_blind"),
                    pl.col("tag_list").alias("_past_tags"),
                ),
                on="track_id", how="left",
            )
        )

    @cached_property
    def _past_tag_set(self) -> pl.DataFrame:
        
        return (
            self._past_tags_long
            .group_by("session_id", "turn_number")
            .agg(pl.col("tag").alias("_past_tag_set"))
        )

    @cached_property
    def _past_tags_long(self) -> pl.DataFrame:
        
        return (
            self._ctx_long.select("session_id", "turn_number", "_past_tags")
            .explode("_past_tags").rename({"_past_tags": "tag"})
            .filter(pl.col("tag").is_not_null())
            .unique(subset=["session_id", "turn_number", "tag"])
        )

    @cached_property
    def _conv_text(self) -> pl.DataFrame | None:
        
        sh = self.session_history_df
        if "user_query" not in sh.columns:
            return None

        exprs = [pl.col("user_query")]
        if "user_thought" in sh.columns:
            exprs.append(pl.col("user_thought"))
        
        if "conversation_goal" in sh.columns and isinstance(
            sh.schema["conversation_goal"], pl.Struct
        ):
            exprs.append(
                pl.col("conversation_goal").struct.field("listener_goal")
                  .alias("goal_text")
            )
            exprs.append(
                pl.col("conversation_goal").struct.field("category")
                  .alias("goal_category")
            )
            exprs.append(
                pl.col("conversation_goal").struct.field("specificity")
                  .alias("goal_specificity")
            )
        ct = sh.select("session_id", "turn_number", *exprs).unique(
            subset=["session_id", "turn_number"]
        )
        
        if "goal_text" in ct.columns:
            ct = ct.with_columns(
                pl.col("goal_text").str.split(" ").list.len()
                  .cast(pl.Int32).alias("goal_description_word_count")
            )
        return ct

    

    @staticmethod
    def _t(tag: str, t0: float) -> float:
        """Log elapsed time of an inner sub-step inside family E.

        Returns the new ``t0`` for the next sub-step.
        """
        now = time.time()
        print(f"    [E] {tag:<28s} {now - t0:>6.2f}s")
        return now

    @staticmethod
    def _tt(family: str, tag: str, t0: float) -> float:
        """Variant of :meth:`_t` parameterised by family letter."""
        now = time.time()
        print(f"    [{family}] {tag:<28s} {now - t0:>6.2f}s")
        return now


    @cached_property
    def _test_track_ids(self) -> list[str]:
        """Track ids present in the blind test-track catalogue (empty when the
        file is absent or disabled)."""
        p = self.test_tracks_path
        if p is None or not Path(p).exists():
            return []
        return pl.read_parquet(p, columns=["track_id"])["track_id"].unique().to_list()

    def _add_C(self, df: pl.DataFrame) -> pl.DataFrame:
        """Family C — track / artist / album popularity stats + test-track flag."""
        t0 = time.time()
        out = df.join(self._track_stats, on="track_id", how="left")
        tt = self._test_track_ids
        flag = pl.col("track_id").is_in(tt) if tt else pl.lit(False)
        out = out.with_columns(flag.cast(pl.Int8).alias("cand_in_test_tracks"))
        self._tt("C", "join track_stats", t0)
        return out

    def _add_K_text_goal(self, df: pl.DataFrame) -> pl.DataFrame:
        
        t0 = time.time()

        
        df = df.with_columns(
            (pl.col("track_release_year") - pl.col("session_mean_year"))
              .alias("release_year_gap")
        )
        t0 = self._tt("K", "release_year_gap", t0)

        if self._conv_text is None:
            self._tt("K", "no conv text — skip goal", t0)
            return df

        num_cols = [
            c for c in ("goal_description_word_count", "goal_category", "goal_specificity")
            if c in self._conv_text.columns
        ]
        df = df.join(
            self._conv_text.select("session_id", "turn_number", *num_cols),
            on=["session_id", "turn_number"], how="left",
        )
        t0 = self._tt("K", "join goal", t0)

        
        onehot: list[pl.Expr] = []
        if "goal_category" in df.columns:
            onehot += [
                (pl.col("goal_category") == lvl).fill_null(False)
                  .cast(pl.Int8).alias(f"goal_cat_{lvl}")
                for lvl in CATEGORY_LEVELS
            ]
        if "goal_specificity" in df.columns:
            onehot += [
                (pl.col("goal_specificity") == lvl).fill_null(False)
                  .cast(pl.Int8).alias(f"goal_spec_{lvl}")
                for lvl in SPECIFICITY_LEVELS
            ]
        if onehot:
            df = df.with_columns(onehot)
        self._tt("K", "goal one-hot", t0)

        return df.drop([c for c in ("goal_category", "goal_specificity") if c in df.columns])

    def _add_G(self, df: pl.DataFrame) -> pl.DataFrame:
        """Family G — user metadata + warm flag."""
        t0 = time.time()
        warm = pl.DataFrame({
            "user_id": list(self.warm_user_ids),
            "is_warm_user": [True] * len(self.warm_user_ids),
        })
        t0 = self._tt("G", "build warm table", t0)
        df = df.join(self._user_meta_clean, on="user_id", how="left")
        t0 = self._tt("G", "join user_meta", t0)
        df = df.join(warm, on="user_id", how="left").with_columns(
            pl.col("is_warm_user").fill_null(False)
        )
        self._tt("G", "join warm flag", t0)
        return df

    def _add_F_I(self, df: pl.DataFrame) -> pl.DataFrame:
        
        t0 = time.time()
        ctx, all_targets = self._session_ctx
        t0 = self._tt("F", "_session_ctx", t0)

        
        ctx_long = self._ctx_long
        t0 = self._tt("F", "ctx_long (cached)", t0)

        
        artist_pop_agg = ctx_long.group_by("session_id", "turn_number").agg(
            pl.col("_past_artist").drop_nulls().n_unique().cast(pl.Int32)
              .alias("session_n_unique_artists"),
            pl.col("_past_pop").mean().alias("session_mean_pop"),
            pl.col("_past_pop").std().alias("session_std_pop"),
            pl.col("_past_pop").max().alias("session_max_pop"),
            pl.col("_past_duration").sum().alias("hist_duration_sum"),
            
            pl.col("_past_year").filter(pl.col("_past_year") > 0).mean()
              .alias("session_mean_year"),
            pl.col("_past_year").filter(pl.col("_past_year") > 0).std()
              .alias("hist_release_year_std"),
            
            pl.col("_past_blind").cast(pl.Int8).mean()
              .alias("frac_history_blind_catalog"),
            pl.col("_past_blind").min().alias("session_all_blind_catalog"),
        )
        t0 = self._tt("F", "artist+pop agg", t0)

        
        tags_long = (
            ctx_long.select("session_id", "turn_number", "_past_tags")
            .explode("_past_tags").rename({"_past_tags": "tag"})
            .filter(pl.col("tag").is_not_null())
        )
        tag_agg = tags_long.group_by("session_id", "turn_number").agg(
            pl.col("tag").n_unique().cast(pl.Int32).alias("session_n_unique_tags"),
        )
        t0 = self._tt("F", "tag agg", t0)

        
        df = (
            df.join(ctx,            on=["session_id", "turn_number"], how="left")
              .join(all_targets,    on="session_id",                  how="left")
              .join(artist_pop_agg, on=["session_id", "turn_number"], how="left")
              .join(tag_agg,        on=["session_id", "turn_number"], how="left")
              .with_columns(
                  pl.col("ctx_track_ids").fill_null([]),
                  pl.col("ctx_track_ids").list.len().alias("session_length_so_far"),
                  (pl.col("turn_number") / pl.col("max_turn")).alias("turn_position_norm"),
                  (pl.col("turn_number") == 8).alias("is_last_turn"),
              )
        )

        
        if "ctx_track_ids" in df.columns:
            df = df.drop("ctx_track_ids")
        self._tt("F", "join + with_columns + drop ctx_track_ids", t0)
        return df

    def _add_B(self, df: pl.DataFrame, cg_names: list[str]) -> pl.DataFrame:
        """Family B — CG-family consensus (behavioural vs semantic)."""
        t0 = time.time()
        beh = [c for c in cg_names if c in BEHAVIORAL_CGS]
        sem = [c for c in cg_names if c in SEMANTIC_CGS]
        
        def ret_in(cgs: list[str]) -> pl.Expr:
            if not cgs:
                return pl.lit(0)
            return sum(
                pl.col(f"rank_{c}").is_not_null().cast(pl.Int32) for c in cgs
            )

        df = df.with_columns(
            ret_in(beh).alias("n_behavioral_retrieving"),
            ret_in(sem).alias("n_semantic_retrieving"),
        )
        t0 = self._tt("B", "family counts", t0)
        df = df.with_columns(
            ((pl.col("n_behavioral_retrieving") > 0) & (pl.col("n_semantic_retrieving") == 0))
                .alias("retrieved_behavioral_only"),
            ((pl.col("n_semantic_retrieving") > 0) & (pl.col("n_behavioral_retrieving") == 0))
                .alias("retrieved_semantic_only"),
            ((pl.col("n_behavioral_retrieving") > 0) & (pl.col("n_semantic_retrieving") > 0))
                .alias("retrieved_both_families"),
        )
        t0 = self._tt("B", "family flags", t0)
        
        ranks = [pl.col(f"rank_{c}").cast(pl.Float64) for c in cg_names]
        sum_ranks = sum(
            pl.when(r.is_not_null()).then(r).otherwise(0.0) for r in ranks
        )
        n_not_null = sum(r.is_not_null().cast(pl.Int32) for r in ranks)
        df = df.with_columns(
            pl.when(n_not_null > 0)
              .then(sum_ranks / n_not_null.cast(pl.Float64))
              .otherwise(None)
              .alias("mean_rank_across_cgs")
        )
        self._tt("B", "mean_rank_across_cgs", t0)

        beh_ranks = [pl.col(f"rank_{c}").cast(pl.Float64) for c in beh]
        sem_ranks = [pl.col(f"rank_{c}").cast(pl.Float64) for c in sem]

        if beh_ranks and sem_ranks:
            sum_beh = sum(pl.when(r.is_not_null()).then(r).otherwise(0.0) for r in beh_ranks)
            n_beh = sum(r.is_not_null().cast(pl.Int32) for r in beh_ranks)
            
            sum_sem = sum(pl.when(r.is_not_null()).then(r).otherwise(0.0) for r in sem_ranks)
            n_sem = sum(r.is_not_null().cast(pl.Int32) for r in sem_ranks)

            df = df.with_columns(
                pl.when(n_beh > 0).then(sum_beh / n_beh.cast(pl.Float64)).otherwise(None).alias("_mean_beh_rank"),
                pl.when(n_sem > 0).then(sum_sem / n_sem.cast(pl.Float64)).otherwise(None).alias("_mean_sem_rank"),
            ).with_columns(
                (pl.col("_mean_beh_rank") - pl.col("_mean_sem_rank")).abs().alias("semantic_behavioral_disagreement")
            ).drop("_mean_beh_rank", "_mean_sem_rank")
            self._tt("B", "semantic_behavioral_disagreement", t0)

        
        df = df.with_columns(
            sum((pl.col(f"rank_{c}") == 1).fill_null(False).cast(pl.Int32)
                for c in cg_names).alias("n_cgs_top1")
        ).with_columns(
            (pl.col("n_cgs_top1") > 0).alias("is_top1_any_cg")
        )
        
        if "fusion_combsum" in df.columns:
            g = ("session_id", "turn_number")
            df = df.with_columns(
                (pl.col("fusion_combsum")
                 / (pl.col("fusion_combsum").max().over(g) + 1e-9))
                  .alias("combsum_frac_of_max"),
                pl.col("fusion_combsum").rank("ordinal", descending=True)
                  .over(g).cast(pl.Int32).alias("combsum_rank_in_group"),
            )
        self._tt("B", "top1 + fusion dominance", t0)
        return df

    def _add_E(self, df: pl.DataFrame) -> pl.DataFrame:
        
        t0 = time.time()
        
        ctx_long = self._ctx_long
        t0 = self._t("ctx_long (cached)", t0)

        
        artist_counts = (
            ctx_long.filter(pl.col("_past_artist").is_not_null())
            .group_by("session_id", "turn_number", "_past_artist")
            .agg(pl.len().cast(pl.Int32).alias("session_artist_count"))
            .rename({"_past_artist": "artist_id_first"})
        )
        df = df.join(
            artist_counts,
            on=["session_id", "turn_number", "artist_id_first"], how="left",
        ).with_columns(
            pl.col("session_artist_count").fill_null(0).cast(pl.Int32)
        )
        t0 = self._t("artist_counts join", t0)

        
        album_counts = (
            ctx_long.filter(pl.col("_past_album").is_not_null())
            .group_by("session_id", "turn_number", "_past_album")
            .agg(pl.len().cast(pl.Int32).alias("session_album_count"))
            .rename({"_past_album": "album_id_first"})
        )
        df = df.join(
            album_counts,
            on=["session_id", "turn_number", "album_id_first"], how="left",
        ).with_columns(
            pl.col("session_album_count").fill_null(0).cast(pl.Int32)
        )
        t0 = self._t("album_counts join", t0)

        
        past_tag_set = self._past_tag_set
        t0 = self._t("past_tag_set (cached)", t0)

        df = (
            df.join(past_tag_set, on=["session_id", "turn_number"], how="left")
              .with_columns(
                  pl.col("tag_list").fill_null([]),
                  pl.col("_past_tag_set").fill_null([]),
              )
              .with_columns(
                  
                  pl.col("tag_list").list.unique().alias("_cand_tag_set"),
              )
              .with_columns(
                  
                  pl.col("_cand_tag_set").list.set_intersection(pl.col("_past_tag_set"))
                    .list.len().cast(pl.Int32).alias("tag_overlap_count"),
              )
              .drop("_past_tag_set", "_cand_tag_set")
        )
        t0 = self._t("tag overlap (list.set_intersection)", t0)

        out = df.with_columns(
            (pl.col("session_artist_count") > 0).alias("cand_artist_in_session"),
            (pl.col("session_album_count") > 0).alias("cand_album_in_session"),
            (pl.col("session_artist_count") / (pl.col("session_length_so_far") + 1))
                .alias("cand_artist_session_frac"),
            (pl.col("tag_list").list.len().fill_null(0).cast(pl.Int32) - pl.col("tag_overlap_count"))
                .alias("cand_tag_novelty"),
            ((pl.col("track_play_count") - pl.col("session_mean_pop")) / 
             (pl.col("session_std_pop") + 1e-6)).alias("session_pop_z_score")
        )
        
        if "tag_list" in out.columns:
            out = out.drop("tag_list")
        self._t("derived flags + drop tag_list", t0)
        return out

    def _add_H_text_only(self, df: pl.DataFrame) -> pl.DataFrame:
        """Family H — text-only features over ``user_query``.

        Vectorised via ``polars.str.contains`` (Rust threads). All
        ``mood_<word>`` columns are Int8 indicator flags. ``cand_artist_in_query``
        / ``cand_track_in_query`` are also Int8.
        """
        t0 = time.time()
        if self._conv_text is None:
            self._tt("H", "no conv text — skip", t0)
            return df

        
        text_cols = ["user_query"]
        if "user_thought" in self._conv_text.columns:
            text_cols.append("user_thought")
        if "goal_text" in self._conv_text.columns:
            text_cols.append("goal_text")
        ct = self._conv_text.select(
            "session_id", "turn_number", *text_cols
        ).with_columns(
            pl.col("user_query").str.to_lowercase().alias("_q_lower"),
            pl.col("user_query").str.len_chars().alias("query_len_chars"),
            pl.col("user_query").str.split(" ").list.len().alias("query_n_tokens"),
        )
        extra_lower = []
        if "user_thought" in ct.columns:
            extra_lower.append(pl.col("user_thought").str.to_lowercase().alias("_th_lower"))
        if "goal_text" in ct.columns:
            extra_lower.append(pl.col("goal_text").str.to_lowercase().alias("_goal_lower"))
        if extra_lower:
            ct = ct.with_columns(extra_lower)
        t0 = self._tt("H", "ct base cols", t0)

        
        ct = ct.with_columns([
            pl.col("_q_lower").str.contains(mood, literal=True)
              .cast(pl.Int8).alias(f"mood_{mood}")
            for mood in self.mood_lexicon
        ])
        t0 = self._tt("H", "mood lexicon hits", t0)

        df = df.join(ct, on=["session_id", "turn_number"], how="left")
        t0 = self._tt("H", "join conv text", t0)

        
        def _mentions(text_low: str, name_col: str) -> pl.Expr:
            return (
                pl.when(pl.col(text_low).is_not_null() & pl.col(name_col).is_not_null())
                  .then(
                      pl.col(text_low).str.contains(
                          pl.col(name_col).str.to_lowercase(), literal=True,
                      ).cast(pl.Int8)
                  ).otherwise(pl.lit(0, dtype=pl.Int8))
            )

        mention_exprs = [
            _mentions("_q_lower", "artist_name_first").alias("cand_artist_in_query"),
            _mentions("_q_lower", "track_name_first").alias("cand_track_in_query"),
        ]
        if "_th_lower" in df.columns:
            mention_exprs.append(
                _mentions("_th_lower", "artist_name_first").alias("cand_artist_in_user_thought")
            )
        if "_goal_lower" in df.columns:
            mention_exprs.append(
                _mentions("_goal_lower", "artist_name_first").alias("cand_artist_in_goal")
            )

        df = df.with_columns(mention_exprs).with_columns(
            
            (pl.col("cand_track_in_query") * 2 + pl.col("cand_artist_in_query")).alias("query_match_score"),
            
            (pl.col("query_len_chars").cast(pl.Float64) /
             (pl.col("track_name_first").str.len_chars().fill_null(0) + 1.0)).alias("query_to_track_len_ratio")
        )

        
        drop_cols = [c for c in (
            "_q_lower", "_th_lower", "_goal_lower",
            "user_query", "user_thought", "goal_text",
        ) if c in df.columns]
        df = df.drop(drop_cols)
        self._tt("H", "artist/track name match + drop text", t0)
        return df
    
    def _add_M_embeddings(self, df: pl.DataFrame) -> pl.DataFrame:
        
        t0 = time.time()
        if self.emb_resources is None:
            self._tt("M", "no emb resources — skip", t0)
            return df
        import os

        from .emb_features import add_embedding_features

        ctx, _ = self._session_ctx
        prev = self._last_turn_ctx.select(
            "session_id", "turn_number", "last_track_id"
        )
        cache_path = self.emb_cache_path
        if cache_path is None:
            df = add_embedding_features(df, self.emb_resources, prev, ctx)
            self._tt("M", "embedding cosine sims", t0)
            return df

        keys = ["session_id", "turn_number", "track_id"]
        cache = pl.read_parquet(cache_path) if os.path.exists(cache_path) else None
        want = df.select(keys).unique()
        
        if cache is not None and want.height:
            probe = add_embedding_features(want.head(1), self.emb_resources, prev, ctx)
            expected = {c for c in probe.columns if c.startswith("emb_")}
            if not expected <= set(cache.columns):
                cache = None
        if cache is not None:
            emb_tbl = want.join(cache, on=keys, how="inner")
            miss = want.join(cache.select(keys), on=keys, how="anti")
        else:
            emb_tbl, miss = None, want
        if emb_tbl is None or miss.height:
            comp = add_embedding_features(miss, self.emb_resources, prev, ctx)
            cols = keys + [c for c in comp.columns if c.startswith("emb_")]
            comp = comp.select(cols)
            emb_tbl = comp if emb_tbl is None else pl.concat([emb_tbl.select(cols), comp])
            merged = comp if cache is None else pl.concat(
                [cache.select(cols), comp]).unique(subset=keys, keep="last")
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            merged.write_parquet(cache_path)
            self._tt("M", f"embedding cosine sims ({miss.height} miss / {want.height})", t0)
        else:
            self._tt("M", "embedding cosine sims (cache hit)", t0)
        return df.join(emb_tbl, on=keys, how="left")

    def _add_J_dynamics(self, df: pl.DataFrame) -> pl.DataFrame:
        """Family J — Turn-to-turn local dynamics (vs t-1)."""
        t0 = time.time()
        last_ctx = self._last_turn_ctx
        t0 = self._tt("J", "last_turn_ctx (cached)", t0)

        df = (
            df.join(last_ctx, on=["session_id", "turn_number"], how="left")
            .with_columns(
                (pl.col("artist_id_first") == pl.col("last_artist_id")).fill_null(False)
                  .cast(pl.Int8).alias("is_same_artist_as_last"),
                (pl.col("album_id_first") == pl.col("last_album_id")).fill_null(False)
                  .cast(pl.Int8).alias("is_same_album_as_last"),
                (pl.col("track_play_count") - pl.col("last_pop")).alias("pop_delta_last_turn"),
                (pl.col("track_release_year") - pl.col("last_year")).abs()
                  .alias("year_abs_diff_last_turn"),
                (pl.col("track_duration") - pl.col("last_duration"))
                  .alias("duration_diff_last_turn"),
                
                ((pl.col("track_pop_bucket") == pl.col("last_pop_bucket"))
                 & (pl.col("track_pop_bucket") >= 0)).fill_null(False)
                  .cast(pl.Int8).alias("pop_bucket_match_last_turn"),
                
            )
            .drop("last_track_id", "last_artist_id", "last_album_id", "last_pop",
                  "last_year", "last_duration", "last_pop_bucket", "last_tags")
        )
        self._tt("J", "join + calc local dynamics", t0)
        return df

    def _add_X_crossfeatures(self, df: pl.DataFrame) -> pl.DataFrame:
        
        t0 = time.time()
        cols = set(df.columns)
        has = lambda c: c in cols          
        mc = "max_calibrated_across_cgs"
        g = ("session_id", "turn_number")
        e: list[pl.Expr] = []

        
        if has("cand_artist_session_frac") and has(mc):
            e.append((pl.col("cand_artist_session_frac") * pl.col(mc))
                     .cast(pl.Float32).alias("artistfrac_x_calib"))
        
        if has("cand_artist_in_session") and has(mc):
            e.append((pl.col("cand_artist_in_session").cast(pl.Float64) * pl.col(mc))
                     .cast(pl.Float32).alias("artistsess_x_calib"))
        
        csig = [c for c in ("is_same_artist_as_last", "is_same_album_as_last",
                            "cand_artist_in_session", "cand_album_in_session") if has(c)]
        if csig:
            e.append(pl.sum_horizontal([pl.col(c).cast(pl.Float64) for c in csig])
                     .cast(pl.Float32).alias("continuity_sum"))

        
        if has(mc) and has("track_pop_pct"):
            e.append((pl.col(mc) * (1.0 - pl.col("track_pop_pct")))
                     .cast(pl.Float32).alias("conf_unpop"))

        
        if has("cand_artist_in_query") and has("emb_qwen_cand_query_cos"):
            e.append((pl.col("cand_artist_in_query").cast(pl.Float64)
                      + pl.col("emb_qwen_cand_query_cos"))
                     .cast(pl.Float32).alias("querymatch_sum"))
        
        if has("emb_qwen_cand_prev_cos") and has("emb_qwen_cand_query_cos"):
            e.append((pl.col("emb_qwen_cand_prev_cos") + pl.col("emb_qwen_cand_query_cos"))
                     .cast(pl.Float32).alias("prev_plus_query"))

        df = df.with_columns(e) if e else df

        
        rel: list[pl.Expr] = []
        
        if has("emb_qwen_cand_prev_cos"):
            rel.append((pl.col("emb_qwen_cand_prev_cos")
                        / (pl.col("emb_qwen_cand_prev_cos").max().over(g) + 1e-9))
                       .cast(pl.Float32).alias("prev_frac_turnmax"))
        
        if has("cand_artist_session_frac"):
            rel.append((pl.col("cand_artist_session_frac")
                        / (pl.col("cand_artist_session_frac").max().over(g) + 1e-9))
                       .cast(pl.Float32).alias("afrac_frac_turnmax"))
        
        if has("emb_qwen_cand_query_cos"):
            rel.append((pl.col("emb_qwen_cand_query_cos")
                        / (pl.col("emb_qwen_cand_query_cos").max().over(g) + 1e-9))
                       .cast(pl.Float32).alias("qcos_frac_turnmax"))
        if rel:
            df = df.with_columns(rel)

        
        gap: list[pl.Expr] = []
        
        if has(mc):
            gap.append((pl.col(mc).max().over(g) - pl.col(mc))
                       .cast(pl.Float32).alias("calib_gap_turnmax"))
       
        if has("fusion_tuned_score"):
            f = "fusion_tuned_score"
            gap.append((pl.col(f).max().over(g) - pl.col(f))
                       .cast(pl.Float32).alias("fusion_gap_turnmax"))
        
        if has("emb_qwen_cand_query_cos"):
            q = "emb_qwen_cand_query_cos"
            gap.append((pl.col(q).max().over(g) - pl.col(q))
                       .cast(pl.Float32).alias("qcos_gap_turnmax"))
            
            gap.append(((pl.col(q) - pl.col(q).mean().over(g))
                        / (pl.col(q).std().over(g) + 1e-9))
                       .cast(pl.Float32).alias("qcos_z_turn"))
        if has("emb_qwen_cand_prev_cos"):
            p = "emb_qwen_cand_prev_cos"
            gap.append((pl.col(p).max().over(g) - pl.col(p))
                       .cast(pl.Float32).alias("prevcos_gap_turnmax"))
        
        if has("emb_qwen_cand_sessmean_cos"):
            s = "emb_qwen_cand_sessmean_cos"
            gap.append((pl.col(s).max().over(g) - pl.col(s))
                       .cast(pl.Float32).alias("qwensess_gap_turnmax"))
        if gap:
            df = df.with_columns(gap)

        
        misc: list[pl.Expr] = []
        
        if has(mc) and has("n_cgs_retrieving"):
            misc.append((pl.col(mc) / (pl.col("n_cgs_retrieving").cast(pl.Float64) + 1.0))
                        .cast(pl.Float32).alias("calib_per_ncg"))
        
        if has("emb_qwen_cand_query_cos") and has("emb_qwen_cand_prev_cos"):
            misc.append((pl.col("emb_qwen_cand_query_cos") * pl.col("emb_qwen_cand_prev_cos"))
                        .cast(pl.Float32).alias("qcos_x_prevcos"))
        
        if misc:
            df = df.with_columns(misc)

        
        mmods = [c for c in ("max_calibrated_across_cgs", "emb_qwen_cand_query_cos",
                             "emb_qwen_cand_prev_cos", "emb_qwen_cand_sessmean_cos")
                 if has(c)]
        if len(mmods) >= 2:
            rkn, zn = [], []
            tmp: list[pl.Expr] = []
            for i, c in enumerate(mmods):
                r, z = f"_mrk{i}", f"_mz{i}"
                tmp.append(pl.col(c).rank("ordinal", descending=True).over(g)
                           .cast(pl.Int32).alias(r))
                tmp.append(((pl.col(c) - pl.col(c).mean().over(g))
                            / (pl.col(c).std().over(g) + 1e-9)).alias(z))
                rkn.append(r); zn.append(z)
            df = df.with_columns(tmp)
            zmean = pl.sum_horizontal(zn) / len(zn)
            zstd = (pl.sum_horizontal([(pl.col(z) - zmean) ** 2 for z in zn]) / len(zn)).sqrt()
            df = df.with_columns([
                
                zmean.cast(pl.Float32).alias("modality_z_mean"),
                
                pl.min_horizontal(rkn).cast(pl.Float32).alias("best_modality_rank"),
                
                pl.sum_horizontal([(pl.col(r) <= 10).cast(pl.Int32) for r in rkn])
                  .cast(pl.Float32).alias("nmod_top10"),
                
                zstd.cast(pl.Float32).alias("modality_z_std"),
            ]).drop(rkn + zn)

        self._tt("X", "cross-module interactions", t0)
        return df

    def _add_BC_blind_catalog(self, df: pl.DataFrame) -> pl.DataFrame:
        
        if "is_blind_catalog" not in df.columns:
            return df
        t0 = time.time()
        cols = set(df.columns)
        has = lambda c: c in cols          
        g = ("session_id", "turn_number")
        bc = pl.col("is_blind_catalog")

        out: list[pl.Expr] = [
            
            bc.cast(pl.Int8).mean().over(g).cast(pl.Float32)
              .alias("group_frac_blind_catalog"),
        ]
        
        for src, name in (("fusion_tuned_score", "rank_among_blind_catalog"),
                          ("max_calibrated_across_cgs", "rank_among_blind_catalog_calib"),
                          ("emb_qwen_cand_query_cos", "rank_among_blind_catalog_qcos")):
            if has(src):
                masked = pl.when(bc).then(pl.col(src)).otherwise(None)
                out.append(masked.rank(descending=True).over(g)
                           .cast(pl.Float32).alias(name))
        df = df.with_columns(out)

        
        inter = [
            (bc.cast(pl.Float64) * pl.col(src)).cast(pl.Float32).alias(name)
            for src, name in (("fusion_tuned_score", "blindcat_x_fusion"),
                              ("emb_qwen_cand_query_cos", "blindcat_x_qcos"),
                              ("query_match_score", "blindcat_x_querymatch"))
            if has(src)
        ]
        if inter:
            df = df.with_columns(inter)
        self._tt("BC", "blind-catalog ranks + interactions", t0)
        return df

    

    def build(self, pool: pl.DataFrame, cg_names: list[str]) -> pl.DataFrame:
        
        t_total = time.time()
        n = len(self.STAGES)
        for i, (name, method_name) in enumerate(self.STAGES, 1):
            t0 = time.time()
            print(f"  [FB {i}/{n}] >>> {name}")
            fn = getattr(self, method_name)
            
            if method_name == "_add_B":
                pool = fn(pool, cg_names)
            else:
                pool = fn(pool)
            print(f"  [FB {i}/{n}] <<< {name:<22s} {pool.shape}  ({time.time() - t0:.1f}s)")
        print(f"  [FB] DONE total={time.time() - t_total:.1f}s  final shape={pool.shape}")
        return pool
