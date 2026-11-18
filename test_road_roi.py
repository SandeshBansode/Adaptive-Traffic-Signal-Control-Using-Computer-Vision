"""
Tests for road_roi.py - the counting zone between the two road-edge lines.
"""

from road_roi import RoadROI, ground_point, load_config, save_roi

# A road narrowing towards the horizon: left edge from (0.4, 0.3) to
# (0.1, 1.0), right edge from (0.6, 0.3) to (0.9, 1.0).
LEFT = [[0.4, 0.3], [0.1, 1.0]]
RIGHT = [[0.6, 0.3], [0.9, 1.0]]
W, H = 1000, 500


def test_point_between_lines_is_inside():
    roi = RoadROI(LEFT, RIGHT)
    assert roi.contains((500, 400), W, H)


def test_points_outside_lines_are_outside():
    roi = RoadROI(LEFT, RIGHT)
    assert not roi.contains((50, 450), W, H)     # left of the road
    assert not roi.contains((950, 450), W, H)    # right of the road
    assert not roi.contains((500, 10), W, H)     # beyond where the edges meet (horizon)


def test_zone_extends_past_the_clicked_points():
    # Edges clicked only between y=0.3 and y=0.6 still bound the whole road
    roi = RoadROI([[0.4, 0.3], [0.25, 0.6]], [[0.6, 0.3], [0.75, 0.6]])
    assert roi.contains((500, 480), W, H)        # near the camera, below the clicks
    assert not roi.contains((60, 480), W, H)     # verge, below the clicks


def test_polygon_is_clipped_to_frame():
    roi = RoadROI(LEFT, RIGHT)
    poly = roi.polygon(W, H)
    assert len(poly) >= 3
    assert poly[:, 0].min() >= 0 and poly[:, 0].max() <= W
    assert poly[:, 1].min() >= 0 and poly[:, 1].max() <= H
    assert len(roi.edge_lines(W, H)) == 2


def test_click_order_does_not_matter():
    flipped = RoadROI(LEFT, [RIGHT[1], RIGHT[0]])
    assert flipped.contains((500, 400), W, H)
    assert not flipped.contains((50, 450), W, H)


def test_ground_point_uses_bottom_centre():
    assert ground_point((100, 50, 200, 150)) == (150.0, 150.0)


def test_box_counts_by_where_it_touches_the_road():
    roi = RoadROI(LEFT, RIGHT)
    # Tall box whose top pokes outside the zone but whose wheels are on the road
    assert roi.contains_rect((450, 0, 550, 400), W, H)
    # Box hanging over the road but standing on the verge
    assert not roi.contains_rect((0, 300, 120, 480), W, H)


def test_scales_with_resolution():
    roi = RoadROI(LEFT, RIGHT)
    assert roi.contains((250, 200), 500, 250)
    assert not roi.contains((25, 225), 500, 250)


def test_save_and_load_roundtrip(tmp_path):
    path = str(tmp_path / "roi.json")
    save_roi("1.mp4", RoadROI(LEFT, RIGHT), path)
    save_roi("2.mp4", RoadROI(RIGHT, LEFT), path)
    config = load_config(path)
    assert set(config) == {"1.mp4", "2.mp4"}
    roi = RoadROI.from_config("1.mp4", config)
    assert roi.contains((500, 400), W, H)


def test_missing_or_bad_config():
    assert RoadROI.from_config("x.mp4", {}) is None
    assert RoadROI.from_config("x.mp4", {"x.mp4": {"left": [[0, 0]]}}) is None
    same_line = {"x.mp4": {"left": [[0, 0], [1, 1]], "right": [[0.2, 0.2], [0.8, 0.8]]}}
    assert RoadROI.from_config("x.mp4", same_line) is None
