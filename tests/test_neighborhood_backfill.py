import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ingestion"))
from neighborhood_backfill import build_polygons, find_district

# Two adjacent 1x1-degree squares, not real Copenhagen geometry — this only
# tests the point-in-polygon logic itself, not the real district shapes.
DISTRICTS = [
    {
        "name": "West Square",
        "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]},
    },
    {
        "name": "East Square",
        "geometry": {"type": "Polygon", "coordinates": [[[1, 0], [2, 0], [2, 1], [1, 1], [1, 0]]]},
    },
]


def test_point_inside_first_polygon_matches():
    polygons = build_polygons(DISTRICTS)
    assert find_district(lon=0.5, lat=0.5, polygons=polygons) == "West Square"


def test_point_inside_second_polygon_matches():
    polygons = build_polygons(DISTRICTS)
    assert find_district(lon=1.5, lat=0.5, polygons=polygons) == "East Square"


def test_point_outside_all_polygons_returns_none():
    polygons = build_polygons(DISTRICTS)
    assert find_district(lon=10.0, lat=10.0, polygons=polygons) is None
