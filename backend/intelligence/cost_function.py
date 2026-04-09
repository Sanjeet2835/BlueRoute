"""
cost_function.py
================
Shared cost scoring for BlueRoute route planning.

Purpose:
  Convert raw graph movement + weather conditions into a single routing cost.

Used by:
  - routing_engine.py (A* search)
  - disruption_detector.py (optional threshold-based checks)

Core idea:
  Lower cost = better route.
  Cost increases when a segment is longer, wind is stronger, waves are higher,
  or the segment enters a risky zone.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

# ----------------------------------------------------------------------------
# Default weights
# ----------------------------------------------------------------------------
# These are intentionally simple and readable. You can tune them later.
#
# The values are not "physics-true". They are decision weights for routing.
#
# Interpretation:
#   distance_cost : base movement cost
#   wind_cost     : penalty per m/s of wind speed
#   wave_cost     : penalty per meter of wave height
#   risk_cost     : fixed penalty for warning/danger zones
#   danger_cost   : extra penalty for danger zones
# ----------------------------------------------------------------------------

DISTANCE_WEIGHT = 1.0
WIND_WEIGHT = 0.8
WAVE_WEIGHT = 1.2
WARNING_RISK_PENALTY = 25.0
DANGER_RISK_PENALTY = 100.0


@dataclass(frozen=True)
class WeatherState:
    """
    Weather information attached to a route node or edge.

    Fields:
        wind_speed_mps: Wind speed in m/s.
        wave_height_m: Significant wave height in meters.
        risk: One of "safe", "warning", "danger".
    """

    wind_speed_mps: float = 0.0
    wave_height_m: float = 0.0
    risk: str = "safe"


# ----------------------------------------------------------------------------
# Basic helpers
# ----------------------------------------------------------------------------

def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return great-circle distance between two lat/lon points in kilometers."""
    r = 6371.0

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ----------------------------------------------------------------------------
# Cost functions
# ----------------------------------------------------------------------------

def movement_cost_km(distance_km: float) -> float:
    """Base cost for moving across an edge."""
    if distance_km < 0:
        raise ValueError("distance_km must be non-negative")
    return DISTANCE_WEIGHT * distance_km


def weather_penalty(weather: Optional[WeatherState]) -> float:
    """
    Penalty added for weather at a node/segment.

    Uses simple linear penalties so the routing engine can remain easy to explain.
    """
    if weather is None:
        return 0.0

    penalty = 0.0
    penalty += WIND_WEIGHT * max(0.0, weather.wind_speed_mps)
    penalty += WAVE_WEIGHT * max(0.0, weather.wave_height_m)

    if weather.risk == "warning":
        penalty += WARNING_RISK_PENALTY
    elif weather.risk == "danger":
        penalty += DANGER_RISK_PENALTY

    return penalty


def segment_cost(
    distance_km: float,
    weather: Optional[WeatherState] = None,
) -> float:
    """
    Final routing cost for one edge.

    Formula:
        cost = base movement cost + weather penalty
    """
    return movement_cost_km(distance_km) + weather_penalty(weather)


def node_cost(
    weather: Optional[WeatherState] = None,
) -> float:
    """
    Cost of visiting a node itself.

    This is useful when you want the route to avoid bad weather even if the edge
    length is short.
    """
    return weather_penalty(weather)


# ----------------------------------------------------------------------------
# Heuristic for A*
# ----------------------------------------------------------------------------

def heuristic_cost_km(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    """
    A* heuristic: straight-line distance only.

    Keep heuristic optimistic so A* remains valid:
      heuristic <= true cost
    """
    return movement_cost_km(haversine_distance_km(lat1, lon1, lat2, lon2))


# ----------------------------------------------------------------------------
# Convenience wrapper
# ----------------------------------------------------------------------------

def total_route_score(
    distance_km: float,
    wind_speed_mps: float = 0.0,
    wave_height_m: float = 0.0,
    risk: str = "safe",
) -> float:
    """
    Small helper for quick experiments or testing.

    Example:
        score = total_route_score(12.4, wind_speed_mps=14, wave_height_m=3.2, risk="warning")
    """
    weather = WeatherState(
        wind_speed_mps=wind_speed_mps,
        wave_height_m=wave_height_m,
        risk=risk,
    )
    return segment_cost(distance_km, weather)


if __name__ == "__main__":
    # Tiny sanity check
    d = 12.0
    weather = WeatherState(wind_speed_mps=16.0, wave_height_m=4.0, risk="warning")
    print("segment_cost:", round(segment_cost(d, weather), 2))
    print("heuristic_cost_km:", round(heuristic_cost_km(25.0, 72.0, 26.0, 73.0), 2))
