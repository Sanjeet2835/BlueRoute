"""
grid_builder.py
================
Builds a sparse ocean waypoint graph for routing.

Nodes:
  - Lat/Lon grid points over defined bbox

Edges:
  - Connections to neighboring nodes (8-directional)

Output:
  - Graph structure usable by A* routing engine
"""

import math
import json
from typing import Dict, List, Tuple

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

BBOX = {
    "lat_min": 10.0,
    "lat_max": 45.0,
    "lon_min": 48.0,
    "lon_max": 148.0,
}

GRID_RESOLUTION_DEG = 1.0   # keep same as ingestion for alignment

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _frange(start: float, stop: float, step: float) -> List[float]:
    vals = []
    v = start
    while v <= stop + 1e-9:
        vals.append(round(v, 6))
        v += step
    return vals


def haversine_distance(lat1, lon1, lat2, lon2) -> float:
    """
    Distance in km between two lat/lon points
    """
    R = 6371  # Earth radius (km)

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ─────────────────────────────────────────────
# Core: Build nodes
# ─────────────────────────────────────────────

def build_nodes() -> List[Tuple[float, float]]:
    """
    Generate grid nodes across bounding box
    """
    lats = _frange(BBOX["lat_min"], BBOX["lat_max"], GRID_RESOLUTION_DEG)
    lons = _frange(BBOX["lon_min"], BBOX["lon_max"], GRID_RESOLUTION_DEG)

    nodes = []
    for lat in lats:
        for lon in lons:
            nodes.append((lat, lon))

    return nodes


# ─────────────────────────────────────────────
# Core: Build edges
# ─────────────────────────────────────────────

def get_neighbors(lat: float, lon: float) -> List[Tuple[float, float]]:
    """
    8-directional connectivity
    """
    step = GRID_RESOLUTION_DEG

    directions = [
        ( step,  0), (-step,  0),
        ( 0,  step), ( 0, -step),
        ( step,  step), ( step, -step),
        (-step,  step), (-step, -step),
    ]

    neighbors = []

    for dlat, dlon in directions:
        nlat = round(lat + dlat, 6)
        nlon = round(lon + dlon, 6)

        if (BBOX["lat_min"] <= nlat <= BBOX["lat_max"] and
            BBOX["lon_min"] <= nlon <= BBOX["lon_max"]):
            neighbors.append((nlat, nlon))

    return neighbors


def build_graph(nodes: List[Tuple[float, float]]) -> Dict:
    """
    Build adjacency list graph
    """
    graph = {}

    for lat, lon in nodes:
        node_id = f"{lat}_{lon}"
        graph[node_id] = []

        neighbors = get_neighbors(lat, lon)

        for nlat, nlon in neighbors:
            neighbor_id = f"{nlat}_{nlon}"

            distance = haversine_distance(lat, lon, nlat, nlon)

            graph[node_id].append({
                "to": neighbor_id,
                "distance": round(distance, 2)
            })

    return graph


# ─────────────────────────────────────────────
# Save graph
# ─────────────────────────────────────────────

def save_graph(graph: Dict, path="data-samples/grid_graph.json"):
    with open(path, "w") as f:
        json.dump(graph, f)
    print(f"Graph saved → {path}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    print("Building ocean grid graph...")

    nodes = build_nodes()
    print(f"Nodes: {len(nodes)}")

    graph = build_graph(nodes)
    print(f"Graph built with {len(graph)} nodes")

    save_graph(graph)


if __name__ == "__main__":
    main()