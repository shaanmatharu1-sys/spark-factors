"""The one list of factors. Order is canonical (used for vectors and the hash).

`lookback` = bar rows read INCLUDING the current one. tests/test_parity.py checks
each is exact: the value from the last `lookback` rows equals the value from the
full history, and `lookback - 1` rows is not enough.
"""
from __future__ import annotations

from spark_factors.families import momentum as M
from spark_factors.families import reversal as R
from spark_factors.families import volatility as V
from spark_factors.families import volume as Q
from spark_factors.spec import FactorSpec


def _news(name):
    def fn(*_):  # computed in bulk by families.sentiment.sentiment_features
        raise RuntimeError(f"{name} is computed by sentiment_features, not per-panel")
    fn.__name__ = name
    return fn


REGISTRY: tuple[FactorSpec, ...] = (
    # momentum / trend
    FactorSpec("mom_1m", "momentum", 22, "log return over 21 sessions", M.mom_1m),
    FactorSpec("mom_3m", "momentum", 64, "log return over 63 sessions", M.mom_3m),
    FactorSpec("mom_6m", "momentum", 127, "log return over 126 sessions", M.mom_6m),
    FactorSpec("mom_12_1", "momentum", 253, "12-1 month momentum", M.mom_12_1),
    FactorSpec("high_52w_prox", "momentum", 252, "close / 252d high - 1", M.high_52w_prox),
    FactorSpec("ma_50_200", "momentum", 200, "SMA50 / SMA200 - 1", M.ma_50_200),
    FactorSpec("macd_hist", "momentum", 161, "MACD(12,26,9) histogram / price", M.macd_hist),
    FactorSpec("adx_14", "momentum", 250, "ADX(14) trend strength", M.adx_14),
    FactorSpec("trend_r2_63", "momentum", 63, "signed R^2 of log price on time", M.trend_r2_63),
    # short-term reversal / oscillators
    FactorSpec("rev_5d", "reversal", 6, "minus 5-session log return", R.rev_5d),
    FactorSpec("gap_1d", "reversal", 2, "overnight gap log(O/C_prev)", R.gap_1d),
    FactorSpec("intraday_ret", "reversal", 1, "session log(C/O) so far", R.intraday_ret),
    FactorSpec("rsi_14", "reversal", 126, "RSI(14)", R.rsi_14),
    FactorSpec("bb_pctb_20", "reversal", 20, "Bollinger %B(20,2)", R.bb_pctb_20),
    FactorSpec("stoch_k_14", "reversal", 14, "stochastic %K(14)", R.stoch_k_14),
    FactorSpec("vwap_dist_20", "reversal", 20, "close / 20-session VWAP - 1", R.vwap_dist_20),
    # volatility / risk
    FactorSpec("vol_21", "volatility", 22, "21d realised vol (ann.)", V.vol_21),
    FactorSpec("vol_63", "volatility", 64, "63d realised vol (ann.)", V.vol_63),
    FactorSpec("vol_ratio_21_63", "volatility", 64, "vol_21 / vol_63", V.vol_ratio_21_63),
    FactorSpec("idio_vol_63", "volatility", 64, "63d residual vol vs SPY", V.idio_vol_63),
    FactorSpec("beta_126", "volatility", 127, "126d beta to SPY", V.beta_126),
    FactorSpec("max_ret_21", "volatility", 22, "max daily return in 21d", V.max_ret_21),
    FactorSpec("skew_63", "volatility", 64, "63d return skewness", V.skew_63),
    FactorSpec("downside_vol_63", "volatility", 64, "63d downside semivol", V.downside_vol_63),
    FactorSpec("atr_pct_14", "volatility", 126, "ATR(14) / price", V.atr_pct_14),
    # volume / liquidity
    FactorSpec("volume_z_63", "volume", 64, "log volume z vs prior 63d", Q.volume_z_63),
    FactorSpec("obv_slope_21", "volume", 21, "OBV slope over 21d / ADV", Q.obv_slope_21),
    FactorSpec("amihud_21", "volume", 22, "log Amihud illiquidity", Q.amihud_21),
    FactorSpec("turnover_trend_5_63", "volume", 63, "log ADV5 / ADV63", Q.turnover_trend_5_63),
    FactorSpec("updown_vol_21", "volume", 22, "log up-volume / down-volume", Q.updown_vol_21),
    FactorSpec("dollar_vol_21", "volume", 21, "log 21d dollar volume", Q.dollar_vol_21),
    # sentiment (news; lookback is in bars, the news windows are in the family module)
    FactorSpec("sent_1d", "sentiment", 1, "relevance-weighted lexicon score, 24h", _news("sent_1d"), uses_news=True),
    FactorSpec("sent_7d", "sentiment", 1, "relevance-weighted lexicon score, 7d", _news("sent_7d"), uses_news=True),
    FactorSpec("news_count_7d", "sentiment", 1, "log1p relevance-weighted article count, 7d", _news("news_count_7d"), uses_news=True),
    FactorSpec("sent_shock_30d", "sentiment", 1, "sent_1d minus 30d mean", _news("sent_shock_30d"), uses_news=True),
)

FACTOR_NAMES: tuple[str, ...] = tuple(f.name for f in REGISTRY)
BY_NAME = {f.name: f for f in REGISTRY}
MAX_LOOKBACK: int = max(f.lookback for f in REGISTRY)

if len(set(FACTOR_NAMES)) != len(FACTOR_NAMES):
    raise RuntimeError("duplicate factor names in REGISTRY")


def families() -> dict:
    out: dict = {}
    for f in REGISTRY:
        out.setdefault(f.family, []).append(f.name)
    return out
