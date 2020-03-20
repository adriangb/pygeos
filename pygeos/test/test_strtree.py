import math
import pygeos
from pygeos import box, points, linestrings, multipoints, buffer, STRtree
import pytest
import numpy as np
from numpy.testing import assert_array_equal
from .common import point, empty, assert_increases_refcount, assert_decreases_refcount


# the distance between 2 points spaced at whole numbers along a diagonal
HALF_UNIT_DIAG = math.sqrt(2) / 2
EPS = 1e-9

predicates = ("intersects", "within", "contains", "overlaps", "crosses", "touches")

@pytest.fixture
def point_tree():
    geoms = pygeos.points(np.arange(10), np.arange(10))
    yield pygeos.STRtree(geoms)


@pytest.fixture
def line_tree():
    x = np.arange(10)
    y = np.arange(10)
    offset = 1
    geoms = pygeos.linestrings(np.array([[x, x + offset], [y, y + offset]]).T)
    yield pygeos.STRtree(geoms)


@pytest.fixture
def poly_tree():
    # create buffers so that midpoint between two buffers intersects
    # each buffer.  NOTE: add EPS to help mitigate rounding errors at midpoint.
    geoms = pygeos.buffer(
        pygeos.points(np.arange(10), np.arange(10)), HALF_UNIT_DIAG + EPS, quadsegs=32
    )
    yield pygeos.STRtree(geoms)


trees = ("point_tree", "line_tree", "poly_tree")


test_geometries_single = (
    points(0, 0),
    points(0.5, 0.5),
    points(2, 2),
    points(0, 0.5),
    points(1, 1),
    points(HALF_UNIT_DIAG + EPS, 0),
    box(0, 0, 1, 1),
    box(0.5, 0.5, 1.5, 1.5),
    box(0, 0, 1.5, 1.5),
    box(0, 0, 1, 1),
    box(3, 3, 6, 6),
    box(5, 5, 15, 15),
    box(0, 0, 2, 2),
    buffer(pygeos.points(3, 3), HALF_UNIT_DIAG + EPS),
    buffer(points(2.5, 2.5), HALF_UNIT_DIAG),
    buffer(points(3, 3), HALF_UNIT_DIAG),
    buffer(points(0, 1), 1),
    buffer(points(3, 3), 1),
    buffer(points(3, 3), 3 * HALF_UNIT_DIAG),
    buffer(points(3, 3), 0.5),
    buffer(points(3, 3), 1),
    buffer(points(2, 1), HALF_UNIT_DIAG),
    multipoints([[5, 5], [7, 7]]),
    multipoints([[5.5, 5], [7, 7]]),
    multipoints([[5, 7], [7, 5]]),
    multipoints([[6.5, 6.5], [7, 7]]),
    multipoints([[5.25, 5.5], [5.25, 5.0]]),
    multipoints([[5, 5], [6, 6]]),
    multipoints([[5, 7], [7, 7]]),
    multipoints([[5, 7], [7, 7], [7, 8]]),
    multipoints([[0, 0], [7, 7], [7, 8]]),
)

test_geometries_bulk = (
    [pygeos.points(0.5, 0.5)],
    [pygeos.points(1, 1)],
    [pygeos.points(1, 1), pygeos.points(-1, -1), pygeos.points(2, 2)],
    [box(0, 0, 1, 1)],
    [box(5, 5, 15, 15)],
    [box(0, 0, 1, 1), box(100, 100, 110, 110), box(5, 5, 15, 15)],
    [pygeos.buffer(pygeos.points(3, 3), 1)],
    [pygeos.multipoints([[5, 7], [7, 5]])],
    [None, empty, pygeos.points(1, 1)],
    [pygeos.points(0, 0)],
    [pygeos.points(0.5, 0.5)],
    [pygeos.points(0, 0.5)],
    [pygeos.points(1, 1)],
    [box(0, 0, 1, 1)],
    [pygeos.buffer(pygeos.points(3, 3), 0.5)],
    [pygeos.multipoints([[5, 7], [7, 5]])],
    [pygeos.points(0.5, 0.5)],
    [pygeos.points(1, 1)],
    [box(0, 0, 1, 1)],
    [box(0, 0, 1, 1), box(100, 100, 110, 110), box(2, 2, 3, 3)],
    [box(0, 0, 1.5, 1.5)],
    [pygeos.buffer(pygeos.points(3, 3), HALF_UNIT_DIAG)],
    [pygeos.buffer(pygeos.points(3, 3), 3 * HALF_UNIT_DIAG)],
    [pygeos.multipoints([[5, 7], [7, 5]])],
)

test_geometries_bulk = test_geometries_bulk + test_geometries_single


def brute_force_nearest_search(geometry, tree, report_distances):
    """Brute force search for nearest by computing all distances.
    """
    left = np.atleast_1d(geometry)
    right = np.atleast_1d(tree.geometries)
    inds_left = []
    inds_right = []
    distances = []
    for ind_left, geom in enumerate(left):
        d = pygeos.distance(geom, right)
        min_dist = np.min(d)
        matches = np.arange(right.shape[0])[d==min_dist]
        inds_left.extend([ind_left] * matches.shape[0])
        inds_right.extend(matches)
        distances.extend(d[d==min_dist])
    if report_distances:
        res = np.array([inds_left, inds_right, distances], dtype=float)
    else:
        res = np.array([inds_left, inds_right], dtype=int)
    return res


def check_nearest(geometry, tree):
    """Compares tree results with brute force search.
    """
    for rep_dist in (True, False):
        matches = tree.nearest(geometry, rep_dist)
        true_matches = brute_force_nearest_search(geometry, tree, rep_dist)
        assert_array_equal(matches, true_matches)

def do_op(geometry, right, predicate):
    if predicate:
        return np.flatnonzero(getattr(pygeos.predicates, predicate)(geometry, right))
    else:
        extent_geo = pygeos.box(*pygeos.bounds(geometry))
        if not pygeos.area(extent_geo): # point or points
            extent_geo = geometry
        extents_right = []
        for geo_right in right:
            extent = pygeos.box(*pygeos.bounds(geo_right))
            if pygeos.area(extent):
                extents_right.append(extent)
            else:
                extents_right.append(geo_right)
        return np.flatnonzero(getattr(pygeos.predicates, "intersects")(extent_geo, extents_right))

def brute_force_query_bulk(geometry, tree, predicate):
    """Brute force search for nearest by checking all geometries.
    """
    left = np.atleast_1d(geometry)
    right = np.atleast_1d(tree.geometries)
    inds_left = []
    inds_right = []
    for ind_left, geom in enumerate(left):
        matches = do_op(geom, right, predicate)
        inds_right.extend(matches)
        inds_left.extend(ind_left for _ in range(matches.size))
    return np.array([inds_left, inds_right], dtype=int).reshape(2, -1)

def brute_force_query(geometry, tree, predicate):
    """Brute force search for nearest by checking all geometries.
    """
    if not geometry or pygeos.is_empty(geometry):
        return None
    right = np.atleast_1d(tree.geometries)
    return do_op(geometry, right, predicate)
    

def check_query(geometry, tree, predicate=None):
    """Compares tree results with brute force search.
    """
    matches = tree.query(geometry, predicate)
    true_matches = brute_force_query(geometry, tree, predicate)
    assert_array_equal(matches, true_matches)

def check_query_bulk(geometry, tree, predicate=None):
    """Compares tree results with brute force search.
    """
    matches = tree.query_bulk(geometry, predicate)
    true_matches = brute_force_query_bulk(geometry, tree, predicate)
    assert_array_equal(matches, true_matches)


def test_init_with_none():
    tree = pygeos.STRtree(np.array([None]))
    assert tree.query(point).size == 0


def test_init_with_no_geometry():
    with pytest.raises(TypeError):
        pygeos.STRtree(np.array(["Not a geometry"], dtype=object))


def test_init_increases_refcount():
    arr = np.array([point])
    with assert_increases_refcount(point):
        _ = pygeos.STRtree(arr)


def test_del_decreases_refcount():
    arr = np.array([point])
    tree = pygeos.STRtree(arr)
    with assert_decreases_refcount(point):
        del tree


def test_flush_geometries():
    arr = pygeos.points(np.arange(10), np.arange(10))
    tree = pygeos.STRtree(arr)
    # Dereference geometries
    arr[:] = None
    import gc

    gc.collect()
    # Still it does not lead to a segfault
    tree.query(point)


def test_len():
    arr = np.array([point, None, point])
    tree = pygeos.STRtree(arr)
    assert len(tree) == 2


def test_geometries_property():
    arr = np.array([point])
    tree = pygeos.STRtree(arr)
    assert arr is tree.geometries


def test_query_no_geom(point_tree):
    with pytest.raises(TypeError):
        point_tree.query("I am not a geometry")


def test_query_none(point_tree):
    assert point_tree.query(None).size == 0


def test_query_empty(point_tree):
    assert point_tree.query(empty).size == 0


@pytest.mark.parametrize(
    "geometry", test_geometries_single
)
@pytest.mark.parametrize(
    "tree", trees,
    ids=trees
)
def test_query(request, tree, geometry):
    tree = request.getfixturevalue(tree)
    check_query(geometry, tree)

def test_query_invalid_predicate(tree):
    with pytest.raises(ValueError):
        tree.query(pygeos.points(1, 1), predicate="bad_predicate")


def test_query_unsupported_predicate(point_tree):
    # valid GEOS binary predicate, but not supported for query
    with pytest.raises(ValueError):
        point_tree.query(pygeos.points(1, 1), predicate="disjoint")


def test_query_tree_with_none():
    tree = pygeos.STRtree(
        [pygeos.Geometry("POINT (0 0)"), None, pygeos.Geometry("POINT (2 2)")]
    )
    assert tree.query(pygeos.points(2, 2), predicate="intersects") == [2]


### predicate == 'intersects'

@pytest.mark.parametrize(
    "geometry", test_geometries_single
)
@pytest.mark.parametrize(
    "predicate", predicates,
    ids=predicates,
)
@pytest.mark.parametrize(
    "tree", trees,
    ids=trees
)
def test_query_intersects_points(request, tree, geometry, predicate):
    tree = request.getfixturevalue(tree)
    check_query(geometry, tree, predicate=predicate)

### Bulk query tests
def test_query_bulk_wrong_dimensions(point_tree):
    with pytest.raises(TypeError, match="Array should be one dimensional"):
        point_tree.query_bulk([[pygeos.points(0.5, 0.5)]])


@pytest.mark.parametrize("geometry", [[], "foo", 1])
def test_query_bulk_wrong_type(point_tree, geometry):
    with pytest.raises(TypeError, match="Array should be of object dtype"):
        point_tree.query_bulk(geometry)


@pytest.mark.parametrize(
    "geometry", test_geometries_bulk
)
@pytest.mark.parametrize(
    "predicate", predicates,
    ids=predicates,
)
@pytest.mark.parametrize(
    "tree", trees,
    ids=trees
)
def test_query_bulk_points(request, tree, geometry, predicate):
    tree = request.getfixturevalue(tree)
    check_query_bulk(geometry, tree, predicate=predicate)

def test_query_invalid_predicate(point_tree):
    with pytest.raises(ValueError):
        point_tree.query_bulk(pygeos.points(1, 1), predicate="bad_predicate")

@pytest.mark.parametrize("geometry", ["I am not a geometry", ["I am not a geometry"]])
def test_nearest_no_geom(point_tree, geometry):
    with pytest.raises(TypeError):
        point_tree.nearest(geometry)

### Nearest queries

@pytest.mark.parametrize("geometry", (None, [None]))
def test_nearest_none(point_tree, geometry):
    check_nearest(geometry, point_tree)


@pytest.mark.parametrize("geometry", (empty, [empty], []))
def test_nearest_empty(point_tree, geometry):
    # make sure dtype is object, we are not checking for that error
    geometry = np.array(geometry, dtype=object)
    check_nearest(geometry, point_tree)


@pytest.mark.parametrize(
    "geometry", test_geometries_single
)
@pytest.mark.parametrize(
    "tree", trees,
    ids=trees
)
def test_nearest_points(request, tree, geometry):
    tree = request.getfixturevalue(tree)
    check_nearest(geometry, tree)

@pytest.mark.parametrize(
    "geometry", test_geometries_bulk
)
@pytest.mark.parametrize(
    "tree", trees,
    ids=trees
)
def test_nearest_points_distances(request, tree, geometry):
    """Tests different usages of `report_distances` param.
    """
    tree = request.getfixturevalue(tree)
    # True
    expected = brute_force_nearest_search(geometry, tree, True)
    assert_array_equal(tree.nearest(geometry, True), expected)
    assert_array_equal(tree.nearest(geometry, report_distances=True), expected)
    assert_array_equal(tree.nearest(geometry=geometry, report_distances=True), expected)
    # False
    expected = brute_force_nearest_search(geometry, tree, False)
    assert_array_equal(tree.nearest(geometry, False), expected)
    assert_array_equal(tree.nearest(geometry, report_distances=False), expected)
    assert_array_equal(tree.nearest(geometry=geometry, report_distances=False), expected)
