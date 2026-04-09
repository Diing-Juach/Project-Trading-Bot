from __future__ import annotations

from typing import Tuple


def get_entry(df, i: int, direction: str) -> Tuple[float, float]:
    #Pulling the live entry and stop from the active zone for this direction
    if direction == "long":
        entry = float(df["demand_zone_entry"].iat[i])
        stop = float(df["demand_zone_stop"].iat[i])
    else:
        entry = float(df["supply_zone_entry"].iat[i])
        stop = float(df["supply_zone_stop"].iat[i])

    return entry, stop
