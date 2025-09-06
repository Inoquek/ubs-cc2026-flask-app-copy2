# routes/investigate.py
import logging
import json
from collections import defaultdict
from flask import request, jsonify
from routes import app

logger = logging.getLogger(__name__)

def calc(network: dict) -> dict:
    """
    Input:  {
      "networkId": "...",
      "network": [{"spy1":"A","spy2":"B"}, ...]
    }
    Output: {
      "networkId": "...",
      "network": [{"spy1":"A","spy2":"B"}, ...]   # edges that remain connected even if removed
    }
    """
    edges = network.get("network", [])
    adj = defaultdict(list)

    # assign an id to each undirected edge
    for eid, edge in enumerate(edges):
        u = edge.get("spy1")
        v = edge.get("spy2")
        if u is None or v is None:
            # skip malformed edges
            continue
        adj[u].append((v, eid))
        adj[v].append((u, eid))

    def dfs(cur: str, target: str, banned_id: int, seen: set) -> bool:
        if cur == target:
            return True
        seen.add(cur)
        for nxt, eid in adj[cur]:
            if eid == banned_id or nxt in seen:
                continue
            if dfs(nxt, target, banned_id, seen):
                return True
        return False

    out = {"networkId": network.get("networkId"), "network": []}

    # For each edge (u, v, id), check if there's an alternate path from u to v when this edge is banned.
    for eid, edge in enumerate(edges):
        u = edge.get("spy1")
        v = edge.get("spy2")
        if u is None or v is None:
            continue
        if dfs(u, v, eid, set()):
            # edge is NOT a bridge (connection still exists without it) -> include it
            out["network"].append({"spy1": u, "spy2": v})

    return out


@app.route("/investigate", methods = ["POST"])
def investigate():
    data = request.get_json(silent=True) or {}

    networks = data.get("networks", [])

    logger.info("Received networks: %d", len(networks))

    result = {"networks": [calc(n) for n in networks]}
    logger.info("investigate result: %s", result)
    return jsonify(result)


