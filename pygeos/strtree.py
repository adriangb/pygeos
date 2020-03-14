from enum import IntEnum
import numpy as np
from pygeos import lib


__all__ = ["STRtree"]


class BinaryPredicate(IntEnum):
    """The enumeration of GEOS binary predicates types"""

    intersects = 1
    within = 2
    contains = 3
    overlaps = 4
    crosses = 5
    touches = 6


VALID_PREDICATES = {e.name for e in BinaryPredicate}


class STRtree:
    """A query-only R-tree created using the Sort-Tile-Recursive (STR)
    algorithm.

    For two-dimensional spatial data. The actual tree will be constructed at the first
    query.

    Parameters
    ----------
    geometries : array_like
    leafsize : int
        the maximum number of child nodes that a node can have

    Examples
    --------
    >>> import pygeos
    >>> tree = pygeos.STRtree(pygeos.points(np.arange(10), np.arange(10)))
    >>> # Query geometries that overlap envelope of input geometries:
    >>> tree.query(pygeos.box(2, 2, 4, 4)).tolist()
    [2, 3, 4]
    >>> # Query geometries that are contained by input geometry:
    >>> tree.query(pygeos.box(2, 2, 4, 4), predicate='contains').tolist()
    [3]
    >>> # Query geometries that overlap envelopes of `geoms`
    >>> tree.query_bulk([pygeos.box(2, 2, 4, 4), pygeos.box(5, 5, 6, 6)])
    (array([0, 0, 0, 1, 1]), array([2, 3, 4, 5, 6]))
    """

    def __init__(self, geometries, leafsize=5):
        self._tree = lib.STRtree(np.asarray(geometries, dtype=np.object), leafsize)

    def __len__(self):
        return self._tree.count

    def query(self, geometry, predicate=None):
        """Return the index of all geometries in the tree with extents that
        intersect the envelope of the input geometry.

        If predicate is provided, a prepared version of the input geometry
        is tested using the predicate function against each item whose
        extent intersects the envelope of the input geometry:
        predicate(geometry, tree_geometry).

        If geometry is None, an empty array is returned.

        Parameters
        ----------
        geometry : Geometry
            The envelope of the geometry is taken automatically for
            querying the tree.
        predicate : {None, 'intersects', 'within', 'contains', 'overlaps', 'crosses', 'touches'}, optional
            The predicate to use for testing geometries from the tree
            that are within the input geometry's envelope.

        Returns
        -------
        ndarray
            Indexes of geometries in tree

        Examples
        --------
        >>> import pygeos
        >>> tree = pygeos.STRtree(pygeos.points(np.arange(10), np.arange(10)))
        >>> tree.query(pygeos.box(1,1, 3,3)).tolist()
        [1, 2, 3]
        >>> # Query geometries that are contained by input geometry
        >>> tree.query(pygeos.box(2, 2, 4, 4), predicate='contains').tolist()
        [3]
        """

        if geometry is None:
            return np.array([], dtype=np.intp)

        if predicate is None:
            predicate = 0

        else:
            if not predicate in VALID_PREDICATES:
                raise ValueError(
                    "Predicate {} is not valid; must be one of {}".format(
                        predicate, ", ".join(VALID_PREDICATES)
                    )
                )

            predicate = BinaryPredicate[predicate].value

        return self._tree.query(geometry, predicate)

    def query_bulk(self, geometries, predicate=None):
        """Returns all combinations of input geometries and geometries in the tree
        where the envelope of each input geometry intersects with the envelope of a
        tree geometry.

        If predicate is provided, a prepared version of each input geometry
        is tested using the predicate function against each item whose
        extent intersects the envelope of the input geometry:
        predicate(geometry, tree_geometry).

        This returns two arrays of equal length, which correspond to the indexes
        of the input geometries and indexes of the tree geometries associated
        each.

        In the context of a spatial join, input geometries are the "left" geometries
        that determine the order of the results, and tree geometries are "right" geometries
        that are joined against the left geometries.  This effectively performs
        an inner join, where only those combinations of geometries that can be joined
        based on envelope overlap or optional predicate are returned.

        Any geometry that is None or empty in the input geometries is omitted from
        the output.

        Parameters
        ----------
        geometry : Geometry
            The envelope of each geometry is taken automatically for
            querying the tree.
        predicate : {None, 'intersects', 'within', 'contains', 'overlaps', 'crosses', 'touches'}, optional
            The predicate to use for testing geometries from the tree
            that are within the input geometry's envelope.

        Returns
        -------
        ndarray, ndarray
            The first array contains input geometry indexes.
            The second array contains tree geometry indexes.

        Examples
        --------
        >>> import pygeos
        >>> tree = pygeos.STRtree(pygeos.points(np.arange(10), np.arange(10)))
        >>> tree.query_bulk([pygeos.box(2, 2, 4, 4), pygeos.box(5, 5, 6, 6)])
        (array([0, 0, 0, 1, 1]), array([2, 3, 4, 5, 6]))
        >>> # Query for geometries that contain tree geometries
        >>> tree.query_bulk([pygeos.box(2, 2, 4, 4), pygeos.box(5, 5, 6, 6)], predicate='contains')
        (array([0]), array([3]))
        """

        if predicate is None:
            predicate = 0

        else:
            if not predicate in VALID_PREDICATES:
                raise ValueError(
                    "Predicate {} is not valid; must be one of {}".format(
                        predicate, ", ".join(VALID_PREDICATES)
                    )
                )

            predicate = BinaryPredicate[predicate].value

        return self._tree.query_bulk(np.asarray(geometries), predicate)

    @property
    def geometries(self):
        """Return the array_like of geometries used to construct the STRtree."""
        return self._tree.geometries
