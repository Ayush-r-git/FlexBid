"""Policies of both levels: rules, MPC and learned networks.

| module | what |
|---|---|
| `rules` | the rule-based bids and dispatches |
| `mpc` | the MPC bid |
| `learned` | trained bids and dispatches, and `make_bid` / `make_dispatch` for any policy description |
| `message` | the message from the bid into the dispatch QP, and the computed message |
| `features` | what the learned policies observe |
| `nets` | the bid MLP settings and the deep-set dispatch network (needs PyTorch) |
| `encoding` | how a learned bid's outputs become a bid |
"""
from .rules import BID_RULES, BID_RULES_CAUTIOUS, DISPATCH_RULES, MENU, bid_rule, dispatch_rule

__all__ = ["BID_RULES", "BID_RULES_CAUTIOUS", "DISPATCH_RULES", "MENU", "bid_rule", "dispatch_rule",
           "make_bid", "make_dispatch", "LearnedBid", "LearnedDispatch", "MPCBid"]


def __getattr__(name):
    # learned policies and MPC are imported on first use, so the rules need only NumPy
    if name in ("make_bid", "make_dispatch", "LearnedBid", "LearnedDispatch"):
        from . import learned
        return getattr(learned, name)
    if name == "MPCBid":
        from .mpc import MPCBid
        return MPCBid
    raise AttributeError(name)
