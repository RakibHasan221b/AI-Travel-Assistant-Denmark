import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ingestion"))
from osm_common import to_row
from validation import validate_row

GOOD_ROW = {
    "osm_id": "node/1",
    "name": "Torvehallerne",
    "category": "restaurant",
    "subcategory": "danish",
    "lat": 55.683,
    "lon": 12.567,
    "address": "Frederiksborggade 21, 1360 Kobenhavn",
    "neighborhood": "Indre By",
    "opening_hours": "Mo-Su 10:00-19:00",
    "osm_tags": '{"amenity": "restaurant"}',
}


def test_valid_row_passes_through_unchanged():
    assert validate_row(GOOD_ROW) == GOOD_ROW


def test_blank_name_rejected():
    row = {**GOOD_ROW, "name": "   "}
    assert validate_row(row) is None


def test_unknown_category_rejected():
    row = {**GOOD_ROW, "category": "spaceport"}
    assert validate_row(row) is None


def test_garbage_coordinates_rejected():
    row = {**GOOD_ROW, "lat": 200.0, "lon": 12.567}
    assert validate_row(row) is None


def test_swapped_lat_lon_rejected():
    # A real bug shape: lat/lon swapped puts the point way outside Denmark
    row = {**GOOD_ROW, "lat": 12.567, "lon": 55.683}
    assert validate_row(row) is None


def test_price_level_out_of_range_rejected():
    row = {**GOOD_ROW, "price_level": 9}
    assert validate_row(row) is None


def test_price_level_none_is_allowed():
    row = {**GOOD_ROW, "price_level": None}
    assert validate_row(row) is not None


def test_to_row_drops_a_row_with_valid_shape_but_bad_category(monkeypatch):
    # amenity=restaurant classifies fine, but forge a case that would only be
    # caught by validation: blank name after tag-driven whitespace stripping.
    tags = {"amenity": "restaurant", "name": "   "}
    assert to_row("node", 42, tags, 55.68, 12.57) is None
