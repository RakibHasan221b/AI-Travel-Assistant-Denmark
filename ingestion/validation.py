"""Data validation gate for OSM ingestion rows before they reach the DB.

Runs inside osm_common.to_row(), so both ingestion paths (bulk Geofabrik and
live Nominatim/Overpass) get the same checks for free — same reason to_row()
itself stays the one shared row-shape function. Rejected rows are logged with
a reason and dropped, same as the existing missing-name/unclassified-tag
drops in to_row() — this just makes the "why" explicit and catches shapes
those two checks don't (garbage coordinates, an unexpected category slipping
through TAG_TO_CATEGORY, an out-of-range price_level).
"""

import logging

from geo import BBOX
from pydantic import BaseModel, ValidationError, field_validator

log = logging.getLogger("validation")

ALLOWED_CATEGORIES = {"restaurant", "cafe", "hotel", "landmark", "bar"}

# Generous slack beyond the pilot bbox (not a strict re-check of in_bbox,
# which already ran before to_row()) — this exists to catch structurally
# broken coordinates, e.g. lat/lon swapped or a decimal-degree parsing bug,
# not to enforce the pilot area boundary a second time.
_COORD_SLACK = 0.5


class PlaceRow(BaseModel):
    osm_id: str
    name: str
    category: str
    subcategory: str | None = None
    lat: float
    lon: float
    address: str | None = None
    neighborhood: str | None = None
    opening_hours: str | None = None
    price_level: int | None = None

    @field_validator("name")
    @classmethod
    def name_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("name is blank after stripping")
        return v

    @field_validator("category")
    @classmethod
    def category_allowed(cls, v: str) -> str:
        if v not in ALLOWED_CATEGORIES:
            raise ValueError(f"category {v!r} not in {sorted(ALLOWED_CATEGORIES)}")
        return v

    @field_validator("lat")
    @classmethod
    def lat_plausible(cls, v: float) -> float:
        s, _w, n, _e = BBOX
        if not (s - _COORD_SLACK <= v <= n + _COORD_SLACK):
            raise ValueError(f"lat {v} implausible for a Copenhagen-area record")
        return v

    @field_validator("lon")
    @classmethod
    def lon_plausible(cls, v: float) -> float:
        _s, w, _n, e = BBOX
        if not (w - _COORD_SLACK <= v <= e + _COORD_SLACK):
            raise ValueError(f"lon {v} implausible for a Copenhagen-area record")
        return v

    @field_validator("price_level")
    @classmethod
    def price_level_range(cls, v: int | None) -> int | None:
        if v is not None and not (0 <= v <= 4):
            raise ValueError(f"price_level {v} out of expected 0-4 range")
        return v


def validate_row(row: dict) -> dict | None:
    """Returns the row unchanged if valid, else logs the reason and returns None."""
    fields = {k: v for k, v in row.items() if k != "osm_tags"}
    try:
        PlaceRow(**fields)
    except ValidationError as e:
        reasons = "; ".join(err["msg"] for err in e.errors())
        log.warning(f"rejected row osm_id={row.get('osm_id')!r} name={row.get('name')!r}: {reasons}")
        return None
    return row
