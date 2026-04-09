"""
routing_engine.py
=================
A* routing engine for BlueRoute / NautiGuard.

Purpose:
  Find the lowest-cost vessel path across a sparse ocean graph while accounting
  for weather risk using the shared cost_function module.

This module is designed to work with:
  - backend/data/grid_builder.py  -> creates the graph / nodes
  - backend/intelligence/cost_function.py -> scores movement + weather
  - weather ingestion data stored in Firestore -> supplies weather at a point

Core idea:
  A* searches from a start node to a goal node.
  The route cost increases when edges are longer and/or weather is worse.

Expected graph format:
  graph = {
      "nodes": {
          "25.0_72.0": {"lat": 25.0, "lon": 72.0},
          ...
      },
      "edges": {
          "25.0_72.0": [
              {"to": "26.0_72.0", "distance_km": 111.19},
              ...
          ],
          ...
      }
  }

The weather lookup callback is optional. If omitted, routing becomes distance-only.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from backend.intelligence.cost_function import (
    WeatherState,
    segment_cost,
    heuristic_cost_km,
    haversine_distance_km,
)


# ----------------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class RouteResult:
    """Final route output from A*."""

    path: List[str]
    total_cost: float
    total_distance_km: float
    expanded_nodes: int


WeatherLookup = Callable[[float, float], Optional[WeatherState]]


# ----------------------------------------------------------------------------
# Node helpers
# ----------------------------------------------------------------------------

def _parse_node_id(node_id: str) -> Tuple[float, float]:
    """Convert 'lat_lon' into (lat, lon)."""
    try:
        lat_str, lon_str = node_id.split("_")
        return float(lat_str), float(lon_str)
    except Exception as exc:
        raise ValueError(f"Invalid node id: {node_id!r}") from exc


def _node_weather(node_id: str, weather_lookup: Optional[WeatherLookup]) -> Optional[WeatherState]:
    """Fetch weather for a graph node if a lookup function is available."""
    if weather_lookup is None:
        return None

    lat, lon = _parse_node_id(node_id)
    return weather_lookup(lat, lon)


# ----------------------------------------------------------------------------
# A* search
# ----------------------------------------------------------------------------

def a_star_route(
    graph: Dict,
    start_node_id: str,
    goal_node_id: str,
    weather_lookup: Optional[WeatherLookup] = None,
) -> RouteResult:
    """
    Find the least-cost path from start_node_id to goal_node_id.

    Parameters
    ----------
    graph:
        Dict with keys "nodes" and "edges".
    start_node_id:
        Graph node ID like '25.0_72.0'.
    goal_node_id:
        Graph node ID like '27.0_74.0'.
    weather_lookup:
        Optional callback: (lat, lon) -> WeatherState

    Returns
    -------
    RouteResult
        Contains the path, total cost, total distance, and explored node count.
    """
    if start_node_id not in graph.get("edges", {}):
        raise KeyError(f"Start node not found in graph edges: {start_node_id}")
    if goal_node_id not in graph.get("nodes", {}):
        raise KeyError(f"Goal node not found in graph nodes: {goal_node_id}")

    if start_node_id == goal_node_id:
        return RouteResult(
            path=[start_node_id],
            total_cost=0.0,
            total_distance_km=0.0,
            expanded_nodes=0,
        )

    open_heap: List[Tuple[float, float, str]] = []
    heapq.heappush(open_heap, (0.0, 0.0, start_node_id))

    came_from: Dict[str, Optional[str]] = {start_node_id: None}
    g_score: Dict[str, float] = {start_node_id: 0.0}
    dist_score: Dict[str, float] = {start_node_id: 0.0}
    closed_set = set()
    expanded_nodes = 0

    goal_lat, goal_lon = _parse_node_id(goal_node_id)

    while open_heap:
        current_f, current_g, current = heapq.heappop(open_heap)

        if current in closed_set:
            continue

        expanded_nodes += 1
        closed_set.add(current)

        if current == goal_node_id:
            path = _reconstruct_path(came_from, current)
            return RouteResult(
                path=path,
                total_cost=round(g_score[current], 2),
                total_distance_km=round(dist_score[current], 2),
                expanded_nodes=expanded_nodes,
            )

        current_lat, current_lon = _parse_node_id(current)
        neighbors = graph.get("edges", {}).get(current, [])

        for edge in neighbors:
            neighbor = edge["to"]
            edge_distance_km = float(edge.get("distance_km", 0.0))

            if neighbor in closed_set:
                continue

            neighbor_lat, neighbor_lon = _parse_node_id(neighbor)
            weather = _node_weather(neighbor, weather_lookup)

            step_cost = segment_cost(edge_distance_km, weather)
            tentative_g = g_score[current] + step_cost
            tentative_distance = dist_score[current] + edge_distance_km

            if tentative_g < g_score.get(neighbor, math.inf):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                dist_score[neighbor] = tentative_distance

                h = heuristic_cost_km(
                    neighbor_lat,
                    neighbor_lon,
                    goal_lat,
                    goal_lon,
                )
                f = tentative_g + h
                heapq.heappush(open_heap, (f, tentative_g, neighbor))

    raise ValueError(f"No route found from {start_node_id} to {goal_node_id}")


# ----------------------------------------------------------------------------
# Path reconstruction
# ----------------------------------------------------------------------------

def _reconstruct_path(came_from: Dict[str, Optional[str]], current: str) -> List[str]:
    path = [current]
    while came_from[current] is not None:
        current = came_from[current]  # type: ignore[assignment]
        path.append(current)
    path.reverse()
    return path


# ----------------------------------------------------------------------------
# Convenience utilities
# ----------------------------------------------------------------------------

def nearest_node_id(
    graph: Dict,
    lat: float,
    lon: float,
) -> str:
    """
    Find the nearest node in the graph to a lat/lon.
    Useful for mapping ports to the grid.
    """
    best_node_id: Optional[str] = None
    best_distance = math.inf

    for node_id, node in graph.get("nodes", {}).items():
        nlat = float(node["lat"])
        nlon = float(node["lon"])
        d = haversine_distance_km(lat, lon, nlat, nlon)
        if d < best_distance:
            best_distance = d
            best_node_id = node_id

    if best_node_id is None:
        raise ValueError("Graph has no nodes")

    return best_node_id


def route_between_points(
    graph: Dict,
    start_lat: float,
    start_lon: float,
    goal_lat: float,
    goal_lon: float,
    weather_lookup: Optional[WeatherLookup] = None,
) -> RouteResult:
    """
    Convenience wrapper for start/goal coordinates instead of node IDs.
    """
    start_node = nearest_node_id(graph, start_lat, start_lon)
    goal_node = nearest_node_id(graph, goal_lat, goal_lon)
    return a_star_route(graph, start_node, goal_node, weather_lookup)


# ----------------------------------------------------------------------------
# Minimal demo
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    # Tiny demo graph so the file can be run directly for sanity checks.
    demo_graph = {
        "nodes": {
            "10.0_48.0": {"lat": 10.0, "lon": 48.0},
            "11.0_48.0": {"lat": 11.0, "lon": 48.0},
            "11.0_49.0": {"lat": 11.0, "lon": 49.0},
        },
        "edges": {
            "10.0_48.0": [
                {"to": "11.0_48.0", "distance_km": 111.19},
                {"to": "11.0_49.0", "distance_km": 157.25},
            ],
            "11.0_48.0": [
                {"to": "11.0_49.0", "distance_km": 111.18},
            ],
            "11.0_49.0": [],
        },
    }

    def demo_weather_lookup(lat: float, lon: float) -> Optional[WeatherState]:
        # Pretend the diagonal point is risky.
        if abs(lat - 11.0) < 1e-9 and abs(lon - 49.0) < 1e-9:
            return WeatherState(wind_speed_mps=18.0, wave_height_m=4.0, risk="warning")
        return WeatherState(wind_speed_mps=4.0, wave_height_m=1.0, risk="safe")

    result = a_star_route(
        graph=demo_graph,
        start_node_id="10.0_48.0",
        goal_node_id="11.0_49.0",
        weather_lookup=demo_weather_lookup,
    )

    print("PATH:", result.path)
    print("TOTAL COST:", result.total_cost)
    print("TOTAL DISTANCE KM:", result.total_distance_km)
    print("EXPANDED NODES:", result.expanded_nodes)
