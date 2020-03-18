#include <float.h>

#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <structmember.h>

#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL pygeos_ARRAY_API

#include <numpy/arrayobject.h>
#include <numpy/ndarraytypes.h>
#include <numpy/npy_3kcompat.h>

#include "strtree.h"
#include "geos.h"
#include "pygeom.h"
#include "kvec.h"

# define BUFFER_QUAD_SEGS 50 // segments for buffer arc in STRtree_nearest

/* GEOS function that takes a prepared geometry and a regular geometry
 * and returns bool value */

typedef char FuncGEOS_YpY_b(void *context, const GEOSPreparedGeometry *a,
                            const GEOSGeometry *b);




/* get predicate function based on ID.  See strtree.py::BinaryPredicate for
 * lookup table of id to function name */

FuncGEOS_YpY_b *get_predicate_func(int predicate_id) {
    switch (predicate_id) {
        case 1: {  // intersects
            return (FuncGEOS_YpY_b *)GEOSPreparedIntersects_r;
        }
        case 2: { // within
            return (FuncGEOS_YpY_b *)GEOSPreparedWithin_r;
        }
        case 3: { // contains
            return (FuncGEOS_YpY_b *)GEOSPreparedContains_r;
        }
        case 4: { // overlaps
            return (FuncGEOS_YpY_b *)GEOSPreparedOverlaps_r;
        }
        case 5: { // crosses
            return (FuncGEOS_YpY_b *)GEOSPreparedCrosses_r;
        }
        case 6: { // touches
            return (FuncGEOS_YpY_b *)GEOSPreparedTouches_r;
        }
        default: { // unknown predicate
            PyErr_SetString(PyExc_ValueError, "Invalid query predicate");
            return NULL;
        }
    }
}



/* Copy values from arr to a new numpy integer array.
 *
 * Parameters
 * ----------
 * arr: dynamic vector array to convert to ndarray
 */

static PyArrayObject *copy_kvec_to_npy(npy_intp_vec *arr)
{
    npy_intp i;
    npy_intp size = kv_size(*arr);

    npy_intp dims[1] = {size};
    // the following raises a compiler warning based on how the macro is defined
    // in numpy.  There doesn't appear to be anything we can do to avoid it.
    PyArrayObject *result = (PyArrayObject *) PyArray_SimpleNew(1, dims, NPY_INTP);
    if (result == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "could not allocate numpy array");
        return NULL;
    }

    for (i = 0; i<size; i++) {
        // assign value into numpy array
        *(npy_intp *)PyArray_GETPTR1(result, i) = kv_A(*arr, i);
    }

    return (PyArrayObject *) result;
}


static void STRtree_dealloc(STRtreeObject *self)
{
    void *context = geos_context[0];
    size_t i, size;
    // free the tree
    if (self->ptr != NULL) { GEOSSTRtree_destroy_r(context, self->ptr); }
    // free the geometries
    size = kv_size(self->_geoms);
    for (i = 0; i < size; i++) {
        Py_XDECREF(kv_A(self->_geoms, i));
    }
    kv_destroy(self->_geoms);
    // free the PyObject
    Py_TYPE(self)->tp_free((PyObject *) self);
}

static PyObject *STRtree_new(PyTypeObject *type, PyObject *args,
                             PyObject *kwds)
{
    int node_capacity;
    PyObject *arr;
    void *tree, *ptr;
    npy_intp n, i, count = 0;
    GEOSGeometry *geom;
    pg_geom_obj_vec _geoms;
    GeometryObject *obj;
    GEOSContextHandle_t context = geos_context[0];

    if (!PyArg_ParseTuple(args, "Oi", &arr, &node_capacity)) {
        return NULL;
    }
    if (!PyArray_Check(arr)) {
        PyErr_SetString(PyExc_TypeError, "Not an ndarray");
        return NULL;
    }
    if (!PyArray_ISOBJECT((PyArrayObject *) arr)) {
        PyErr_SetString(PyExc_TypeError, "Array should be of object dtype");
        return NULL;
    }
    if (PyArray_NDIM((PyArrayObject *) arr) != 1) {
        PyErr_SetString(PyExc_TypeError, "Array should be one dimensional");
        return NULL;
    }

    tree = GEOSSTRtree_create_r(context, (size_t) node_capacity);
    if (tree == NULL) {
        return NULL;
    }

    n = PyArray_SIZE((PyArrayObject *) arr);

    kv_init(_geoms);
    for(i = 0; i < n; i++) {
        /* get the geometry */
        ptr = PyArray_GETPTR1((PyArrayObject *) arr, i);
        obj = *(GeometryObject **) ptr;
        /* fail and cleanup incase obj was no geometry */
        if (!get_geom(obj, &geom)) {
            GEOSSTRtree_destroy_r(context, tree);
            // free the geometries
            count = kv_size(_geoms);
            for (i = 0; i < count; i++) { Py_XDECREF(kv_A(_geoms, i)); }
            kv_destroy(_geoms);
            return NULL;
        }
        /* skip incase obj was None */
        if (geom == NULL) {
            kv_push(GeometryObject *, _geoms, NULL);
        } else {
        /* perform the insert */
            Py_INCREF(obj);
            kv_push(GeometryObject *, _geoms, obj);
            count++;
            GEOSSTRtree_insert_r(context, tree, geom, (void *) i );
        }
    }

    STRtreeObject *self = (STRtreeObject *) type->tp_alloc(type, 0);
    if (self == NULL) {
        GEOSSTRtree_destroy_r(context, tree);
        return NULL;
    }
    self->ptr = tree;
    self->count = count;
    self->_geoms = _geoms;
    return (PyObject *) self;
}


/* Callback called by strtree_query with the index of each intersecting geometry
 * and a dynamic vector to push that index onto.
 *
 * Parameters
 * ----------
 * item: index of intersected geometry in the tree
 * user_data: pointer to dynamic vector; index is pushed onto this vector
 * */

void query_callback(void *item, void *user_data)
{
    kv_push(npy_intp, *(npy_intp_vec *)user_data, (npy_intp) item);
}

/* Evaluate the predicate function against a prepared version of geom
 * for each geometry in the tree specified by indexes in out_indexes.
 * out_indexes is updated in place with the indexes of the geometries in the
 * tree that meet the predicate.
 *
 * Parameters
 * ----------
 * predicate_func: pointer to a prepared predicate function, e.g., GEOSPreparedIntersects_r
 * geom: input geometry to prepare and test against each geometry in the tree specified by
 *       in_indexes.
 * tree_geometries: pointer to ndarray of all geometries in the tree
 * in_indexes: dynamic vector of indexes of tree geometries that have overlapping envelopes
 *             with envelope of input geometry.
 * out_indexes: dynamic vector of indexes of tree geometries that meet predicate function.
 *
 * Returns the number of geometries that met the predicate or -1 in case of error.
 * */

static int evaluate_predicate(FuncGEOS_YpY_b *predicate_func,
                              GEOSGeometry *geom,
                              pg_geom_obj_vec *tree_geometries,
                              npy_intp_vec *in_indexes,
                              npy_intp_vec *out_indexes)
{
    GEOSContextHandle_t context = geos_context[0];
    GeometryObject *pg_geom;
    GEOSGeometry *target_geom;
    const GEOSPreparedGeometry *prepared_geom;
    npy_intp i, size, index, count = 0;

    // Create prepared geometry
    prepared_geom = GEOSPrepare_r(context, geom);
    if (prepared_geom == NULL) {
        return -1;
    }

    size = kv_size(*in_indexes);
    for (i = 0; i < size; i++) {
        // get index for right geometries from in_indexes
        index = kv_A(*in_indexes, i);

        // get GEOS geometry from pygeos geometry at index in tree geometries
        pg_geom = kv_A(*tree_geometries, index);
        if (pg_geom == NULL) { continue; }
        get_geom((GeometryObject *) pg_geom, &target_geom);

        // keep the index value if it passes the predicate
        if (predicate_func(context, prepared_geom, target_geom)) {
            kv_push(npy_intp, *out_indexes, index);
            count++;
        }
    }

    GEOSPreparedGeom_destroy_r(context, prepared_geom);

    return count;
}

/* Query the tree based on input geometry and predicate function.
 * The index of each geometry in the tree whose envelope intersects the
 * envelope of the input geometry is returned by default.
 * If predicate function is provided, only the index of those geometries that
 * satisfy the predicate function are returned.
 *
 * args must be:
 * - pygeos geometry object
 * - predicate id (see strtree.py for list of ids)
 * */

static PyObject *STRtree_query(STRtreeObject *self, PyObject *args) {
    GEOSContextHandle_t context = geos_context[0];
    GeometryObject *geometry;
    int predicate_id = 0; // default no predicate
    GEOSGeometry *geom;
    npy_intp_vec query_indexes, predicate_indexes; // Resizable array for matches for each geometry
    npy_intp count;
    FuncGEOS_YpY_b *predicate_func;
    PyArrayObject *result;

    if (self->ptr == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Tree is uninitialized");
        return NULL;
    }

    if (!PyArg_ParseTuple(args, "O!i", &GeometryType, &geometry, &predicate_id)){
        return NULL;
    }

    if (!get_geom(geometry, &geom)) {
        PyErr_SetString(PyExc_TypeError, "Invalid geometry");
        return NULL;
    }

    if (self->count == 0) {
        npy_intp dims[1] = {0};
        return PyArray_SimpleNew(1, dims, NPY_INTP);
    }

    // query the tree for indices of geometries in the tree with
    // envelopes that intersect the geometry.
    kv_init(query_indexes);
    if (geom != NULL && !GEOSisEmpty_r(context, geom)) {
        GEOSSTRtree_query_r(context, self->ptr, geom, query_callback, &query_indexes);
    }

    if (predicate_id == 0 || kv_size(query_indexes) == 0) {
        // No predicate function provided, return all geometry indexes from
        // query.  If array is empty, return an empty numpy array
        result = copy_kvec_to_npy(&query_indexes);
        kv_destroy(query_indexes);
        return (PyObject *) result;
    }

    predicate_func = get_predicate_func(predicate_id);
    if (predicate_func == NULL) {
        kv_destroy(query_indexes);
        return NULL;
    }

    kv_init(predicate_indexes);
    count = evaluate_predicate(predicate_func, geom, &self->_geoms, &query_indexes, &predicate_indexes);
    if (count == -1) {
        // error performing predicate
        kv_destroy(query_indexes);
        kv_destroy(predicate_indexes);
        return NULL;
    }

    result = copy_kvec_to_npy(&predicate_indexes);

    kv_destroy(query_indexes);
    kv_destroy(predicate_indexes);

    return (PyObject *) result;
}

/* Query the tree based on input geometries and predicate function.
 * The index of each geometry in the tree whose envelope intersects the
 * envelope of the input geometry is returned by default.
 * If predicate function is provided, only the index of those geometries that
 * satisfy the predicate function are returned.
 * Returns two arrays of equal length: first is indexes of the source geometries
 * and second is indexes of tree geometries that meet the above conditions.
 *
 * args must be:
 * - ndarray of pygeos geometries
 * - predicate id (see strtree.py for list of ids)
 *
 * */

static PyObject *STRtree_query_bulk(STRtreeObject *self, PyObject *args) {
    GEOSContextHandle_t context = geos_context[0];
    PyObject *arr;
    PyArrayObject *pg_geoms;
    GeometryObject *pg_geom;
    int predicate_id = 0; // default no predicate
    GEOSGeometry *geom;
    npy_intp_vec query_indexes, src_indexes, target_indexes;
    npy_intp i, j, n, size;
    FuncGEOS_YpY_b *predicate_func = NULL;
    PyArrayObject *result;

    if (self->ptr == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Tree is uninitialized");
        return NULL;
    }

    if (!PyArg_ParseTuple(args, "Oi", &arr, &predicate_id)) {
        return NULL;
    }

    if (!PyArray_Check(arr)) {
        PyErr_SetString(PyExc_TypeError, "Not an ndarray");
        return NULL;
    }

    pg_geoms = (PyArrayObject *) arr;
    if (!PyArray_ISOBJECT(pg_geoms)) {
        PyErr_SetString(PyExc_TypeError, "Array should be of object dtype");
        return NULL;
    }

    if (PyArray_NDIM(pg_geoms) != 1) {
        PyErr_SetString(PyExc_TypeError, "Array should be one dimensional");
        return NULL;
    }

    if (predicate_id!=0) {
        predicate_func = get_predicate_func(predicate_id);
        if (predicate_func == NULL) {
            return NULL;
        }
    }

    n = PyArray_SIZE(pg_geoms);

    if (self->count == 0 || n == 0) {
        npy_intp dims[2] = {2, 0};
        return PyArray_SimpleNew(2, dims, NPY_INTP);
    }

    kv_init(src_indexes);
    kv_init(target_indexes);

    for(i = 0; i < n; i++) {
        // get pygeos geometry from input geometry array
        pg_geom = *(GeometryObject **) PyArray_GETPTR1(pg_geoms, i);
        if (!get_geom(pg_geom, &geom)) {
            PyErr_SetString(PyExc_TypeError, "Invalid geometry");
            return NULL;
        }
        if (geom == NULL || GEOSisEmpty_r(context, geom)) {
            continue;
        }

        kv_init(query_indexes);
        GEOSSTRtree_query_r(context, self->ptr, geom, query_callback, &query_indexes);

        if (kv_size(query_indexes) == 0) {
            // no target geoms in query window, skip this source geom
            kv_destroy(query_indexes);
            continue;
        }

        if (predicate_id == 0) {
            // no predicate, push results directly onto target_indexes
            size = kv_size(query_indexes);
            for (j = 0; j < size; j++) {
                kv_push(npy_intp, src_indexes, i);
                kv_push(npy_intp, target_indexes, kv_A(query_indexes, j));
            }
        } else {
            // this pushes directly onto target_indexes
            size = evaluate_predicate(predicate_func, geom,&self->_geoms,
                                      &query_indexes, &target_indexes);

            if (size == -1) {
                PyErr_SetString(PyExc_TypeError, "Error evaluating predicate function");
                kv_destroy(query_indexes);
                kv_destroy(src_indexes);
                kv_destroy(target_indexes);
                return NULL;
            }

            for (j = 0; j < size; j++) {
                kv_push(npy_intp, src_indexes, i);
            }
        }

        kv_destroy(query_indexes);
    }

    size = kv_size(src_indexes);
    npy_intp dims[2] = {2, size};

    // the following raises a compiler warning based on how the macro is defined
    // in numpy.  There doesn't appear to be anything we can do to avoid it.
    result = (PyArrayObject *) PyArray_SimpleNew(2, dims, NPY_INTP);
    if (result == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "could not allocate numpy array");
        return NULL;
    }

    for (i = 0; i<size; i++) {
        // assign value into numpy arrays
        *(npy_intp *)PyArray_GETPTR2(result, 0, i) = kv_A(src_indexes, i);
        *(npy_intp *)PyArray_GETPTR2(result, 1, i) = kv_A(target_indexes, i);
    }

    kv_destroy(src_indexes);
    kv_destroy(target_indexes);
    return (PyObject *) result;
}

/* Calculate the distance between items in the tree and the src_geom.
 *
 * Parameters
 * ----------
 * target_index: index of geometry in tree
 * src_geom: input geometry to compare distance against
 * distance: pointer to distance that gets updated in this function
 * tree_geometries: dynamic vector of geometries in the tree
 * */

int distance_callback(const void *target_index, const void *src_geom,
                      double *distance, pg_geom_obj_vec *tree_geometries)
{
    GEOSContextHandle_t context = geos_context[0];

    GeometryObject *pg_geom;
    GEOSGeometry *target_geom;

    pg_geom = kv_A(*tree_geometries, (npy_intp)target_index);
    get_geom((GeometryObject *) pg_geom, &target_geom);

    // returns error code or 0
    return GEOSDistance_r(context, src_geom, target_geom, distance);
}


/* Find the nearest item in the tree to each input geometry.
 * Returns empty array of shape (2, 0) if tree is empty. */

static PyObject *STRtree_nearest(STRtreeObject *self, PyObject *args, PyObject *kwargs) {
    GEOSContextHandle_t context = geos_context[0];
    PyObject *arr;
    PyArrayObject *pg_geoms;
    GeometryObject *pg_geom;
    GEOSGeometry *geom, *geom_buffered, *first_match;
    npy_intp_vec src_indexes, nearest_indexes;
    npy_intp_vec indiv_src_indexes, indiv_nearest_indexes, indiv_equidist_indexes;
    npy_double_vec distances, indiv_distances;
    npy_intp i, j, n, m, size, nearest, new_match_index;
    PyArrayObject *result;
    double min_dist;
    double distance;
    int rep_dist; // flag for reporting distances, default to False

    static char* argnames[] = {"geometry", "report_distances", NULL};

    // validate inputs
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|p", argnames, &arr, &rep_dist)) {
        PyErr_SetString(PyExc_TypeError, "Invalid arguments received");
        return NULL;
    }

    if (self->ptr == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Tree is uninitialized");
        return NULL;
    }

    if (!PyArray_Check(arr)) {
        PyErr_SetString(PyExc_TypeError, "Not an ndarray");
        return NULL;
    }

    pg_geoms = (PyArrayObject *) arr;
    if (!PyArray_ISOBJECT(pg_geoms)) {
        PyErr_SetString(PyExc_TypeError, "Array should be of object dtype");
        return NULL;
    }

    if (PyArray_NDIM(pg_geoms) != 1) {
        PyErr_SetString(PyExc_TypeError, "Array should be one dimensional");
        return NULL;
    }

    if (self->count == 0) {
        // empty tree
        npy_intp dims[2] = {2, 0};
        return PyArray_SimpleNew(2, dims, NPY_INTP);
    }

    // ensure all arrays are initialized, even for zero length pg_geoms
    kv_init(src_indexes);
    kv_init(nearest_indexes);
    kv_init(distances);
    kv_init(indiv_src_indexes);
    kv_init(indiv_nearest_indexes);
    kv_init(indiv_equidist_indexes);
    kv_init(indiv_distances);

    // loop over pg_geoms and find matches for each source geometry
    n = PyArray_SIZE(pg_geoms);
    for(i = 0; i < n; i++) {
        
        // re-initialize arrays used by each geom
        kv_init(indiv_src_indexes);
        kv_init(indiv_nearest_indexes);
        kv_init(indiv_distances);
        kv_init(indiv_equidist_indexes);

        // get pygeos geometry from input geometry array
        pg_geom = *(GeometryObject **) PyArray_GETPTR1(pg_geoms, i);
        if (!get_geom(pg_geom, &geom)) {
            PyErr_SetString(PyExc_TypeError, "Invalid geometry");
            return NULL;
        }
        if (geom == NULL || GEOSisEmpty_r(context, geom)) {
            continue;
        }

        // find index for nearest geometry (arbitrary in case of multiple matches)
        nearest = (npy_intp)GEOSSTRtree_nearest_generic_r(context, self->ptr,
                                                          geom, geom,
                                                          distance_callback,
                                                          &self->_geoms);
        
        // get distance to first match
        distance_callback(nearest, geom, &distance, &self->_geoms);
        // make that the minimum distance for all future matches
        min_dist = distance;
        
        // look for other equidistant matches
        // first get the geometry for that first match
        pg_geom = kv_A(self->_geoms, nearest);
        // then query the tree for other geometries that intersect the first match
        get_geom((GeometryObject *) pg_geom, &first_match);
        //create an geometry with radius min_dist
        // add a small buffer to ensure we are not zero buffering
        if (min_dist == 0) {
            geom_buffered = geom; // looking for intersections only
        }
        else {
            geom_buffered = GEOSBuffer_r(context, geom, min_dist, BUFFER_QUAD_SEGS);
        }
        GEOSSTRtree_query_r(context, self->ptr, geom_buffered, query_callback, &indiv_equidist_indexes);

        // loop over these new matches and check if they are equidistant
        m = kv_size(indiv_equidist_indexes);
        for(j = 0; j < m; j++) {
            new_match_index = kv_A(indiv_equidist_indexes, j);

            distance_callback(new_match_index, geom, &distance, &self->_geoms);

            if (distance > min_dist) {
                // not equidistant, skip
                continue;
            }

            kv_push(npy_intp, indiv_src_indexes, i);
            kv_push(npy_intp, indiv_nearest_indexes, new_match_index);
            if (rep_dist) {
                kv_push(npy_double, indiv_distances, distance);
            }
        }

        // now add all matches to final results arrays
        m = kv_size(indiv_src_indexes);
        for(j = 0; j < m; j++) {
            kv_push(npy_intp, src_indexes,  kv_A(indiv_src_indexes, j));
            kv_push(npy_intp, nearest_indexes, kv_A(indiv_nearest_indexes, j));
            if (rep_dist) {
                kv_push(npy_double, distances, kv_A(indiv_distances, j));
            }
        }
    }

    // deallocate
    kv_destroy(indiv_src_indexes);
    kv_destroy(indiv_nearest_indexes);
    kv_destroy(indiv_equidist_indexes);
    kv_destroy(indiv_distances);

    // get final size of output
    size = kv_size(src_indexes);

    // the following raises a compiler warning based on how the macro is defined
    // in numpy.  There doesn't appear to be anything we can do to avoid it.
    npy_intp dims[2] = {2, size};
    if (rep_dist) {
        dims[0] = 3;
        result = (PyArrayObject *) PyArray_SimpleNew(2, dims, NPY_FLOAT64);
    }
    else {
        result = (PyArrayObject *) PyArray_SimpleNew(2, dims, NPY_INTP);
    }
    
    if (result == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "could not allocate numpy array");
        return NULL;
    }

    // convert results arrays to numpy python objects
    for (i = 0; i < size; i++) {
        // assign value into numpy arrays
        if (rep_dist) {
            *(npy_double *)PyArray_GETPTR2(result, 0, i) = kv_A(src_indexes, i);
            *(npy_double *)PyArray_GETPTR2(result, 1, i) = kv_A(nearest_indexes, i);
            *(npy_double *)PyArray_GETPTR2(result, 2, i) = kv_A(distances, i);
        }
        else {
            *(npy_intp *)PyArray_GETPTR2(result, 0, i) = kv_A(src_indexes, i);
            *(npy_intp *)PyArray_GETPTR2(result, 1, i) = kv_A(nearest_indexes, i);
        }
    }

    // deallocate
    kv_destroy(src_indexes);
    kv_destroy(nearest_indexes);
    kv_destroy(distances);

    // done
    return (PyObject *) result;
}

static PyMemberDef STRtree_members[] = {
    {"_ptr", T_PYSSIZET, offsetof(STRtreeObject, ptr), READONLY, "Pointer to GEOSSTRtree"},
    {"count", T_LONG, offsetof(STRtreeObject, count), READONLY, "The number of geometries inside the tree"},
    {NULL}  /* Sentinel */
};

static PyMethodDef STRtree_methods[] = {
    {"query", (PyCFunction) STRtree_query, METH_VARARGS,
     "Queries the index for all items whose extents intersect the given search geometry, and optionally tests them "
     "against predicate function if provided. "
    },
    {"query_bulk", (PyCFunction) STRtree_query_bulk, METH_VARARGS,
     "Queries the index for all items whose extents intersect the given search geometries, and optionally tests them "
     "against predicate function if provided. "
    },
    {"nearest", (PyCFunction) STRtree_nearest, METH_VARARGS | METH_KEYWORDS,
     "Queries the index for the nearest item to each of the given search geometries"
    },
    {NULL}  /* Sentinel */
};

PyTypeObject STRtreeType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "pygeos.lib.STRtree",
    .tp_doc = "A query-only R-tree created using the Sort-Tile-Recursive (STR) algorithm.",
    .tp_basicsize = sizeof(STRtreeObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = STRtree_new,
    .tp_dealloc = (destructor) STRtree_dealloc,
    .tp_members = STRtree_members,
    .tp_methods = STRtree_methods
};


int init_strtree_type(PyObject *m)
{
    if (PyType_Ready(&STRtreeType) < 0) {
        return -1;
    }

    Py_INCREF(&STRtreeType);
    PyModule_AddObject(m, "STRtree", (PyObject *) &STRtreeType);
    return 0;
}
