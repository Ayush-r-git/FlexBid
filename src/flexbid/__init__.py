"""FlexBid: an EV fleet that bids its charging flexibility into a real-time power market.

    import flexbid
    cfg = flexbid.make_config("hub-6h", market="lead", market_lead_min=60)
    result = flexbid.run_episode(cfg, seed=200, bid="fixed:8", dispatch="qp")
    print(result["total"])        # $/day
"""
from .episode import evaluate, run_episode
from .scenarios import FLEETS, MARKETS, make_config
from .sim import SimConfig, Simulator

__all__ = ["make_config", "FLEETS", "MARKETS", "SimConfig", "Simulator", "run_episode", "evaluate"]
__version__ = "0.1.0"
