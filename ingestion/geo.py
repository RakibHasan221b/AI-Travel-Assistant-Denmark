"""Shared geography constants — pulled out of osm_common so validation.py can
depend on BBOX without a circular import (osm_common -> validation -> osm_common).
"""

# Central Copenhagen pilot area (Indre By, Vesterbro, Norrebro, Frederiksberg core) —
# south, west, north, east. Narrower than the full municipality on purpose; widen
# once there's a reason to (the bulk path can easily cover more once this works).
BBOX = (55.66, 12.52, 55.71, 12.60)


def in_bbox(lat: float, lon: float) -> bool:
    s, w, n, e = BBOX
    return s <= lat <= n and w <= lon <= e
