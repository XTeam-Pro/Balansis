/* Copyright (c) 2024-2026 Andrey Tikhonov (XTeam-Pro).
 * AGPL-3.0-only OR commercial license; see LICENSING.md. */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <string.h>
#include <stdint.h>
#include "neumaier.h"
#include "exact_dot.h"

static int get_double_buffer(PyObject *input, Py_buffer *view)
{
    if (PyObject_GetBuffer(input, view, PyBUF_FORMAT | PyBUF_STRIDES) < 0)
        return -1;
    int native_double = view->format &&
        (!strcmp(view->format, "d") || !strcmp(view->format, "@d") ||
         !strcmp(view->format, "=d"));
    if (view->ndim != 1 || view->itemsize != sizeof(double) || !native_double ||
        !PyBuffer_IsContiguous(view, 'C') || view->suboffsets != NULL ||
        view->len < 0 || view->len % sizeof(double) != 0) {
        PyBuffer_Release(view);
        PyErr_SetString(PyExc_ValueError,
            "expected a one-dimensional contiguous native float64 buffer");
        return -1;
    }
    return 0;
}

static PyObject *neumaier_sum(PyObject *self, PyObject *input)
{
    (void)self;
    Py_buffer view;
    if (get_double_buffer(input, &view) < 0) return NULL;
    double result, compensation;
    /* Retain the GIL: no borrowed writable buffer is read after releasing it.
     * Callers must still exclude writes by native threads/external processes. */
    enum balansis_sum_status status = balansis_neumaier_sum(
        view.buf, (size_t)(view.len / sizeof(double)), &result, &compensation);
    PyBuffer_Release(&view);
    if (status == BALANSIS_SUM_NONFINITE) {
        PyErr_SetString(PyExc_ValueError, "sum_array requires finite values");
        return NULL;
    }
    if (status == BALANSIS_SUM_OVERFLOW) {
        PyErr_SetString(PyExc_OverflowError, "compensated sum overflowed float64");
        return NULL;
    }
    return Py_BuildValue("(dd)", result, compensation);
}

static PyObject *exact_dot(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *left, *right;
    if (!PyArg_ParseTuple(args, "OO:exact_dot", &left, &right)) return NULL;
    Py_buffer a, b;
    if (get_double_buffer(left, &a) < 0) return NULL;
    if (get_double_buffer(right, &b) < 0) {
        PyBuffer_Release(&a);
        return NULL;
    }
    if (a.len != b.len) {
        PyBuffer_Release(&a);
        PyBuffer_Release(&b);
        PyErr_SetString(PyExc_ValueError, "dot_array requires equal lengths");
        return NULL;
    }
    double result;
    enum balansis_sum_status status = balansis_exact_dot(
        a.buf, b.buf, (size_t)(a.len / sizeof(double)), &result);
    PyBuffer_Release(&a);
    PyBuffer_Release(&b);
    if (status != BALANSIS_SUM_OK) {
        PyErr_SetString(status == BALANSIS_SUM_NONFINITE ? PyExc_ValueError : PyExc_OverflowError,
            status == BALANSIS_SUM_NONFINITE ? "dot_array requires finite values" :
            "rounded dot product overflowed float64");
        return NULL;
    }
    return PyFloat_FromDouble(result);
}

static PyMethodDef methods[] = {
    {"neumaier_sum", neumaier_sum, METH_O,
     "Return (sum, signed correction) for a contiguous native float64 buffer."},
    {"exact_dot", exact_dot, METH_VARARGS,
     "Exact dot product of finite binary64 buffers, rounded once to nearest-even."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "_balansis_kernels",
    "Strict binary64 sum and exact dot product. Backward-compatible kernel API 1.",
    -1, methods, NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC PyInit__balansis_kernels(void)
{
    double one = 1.0;
    uint64_t bits;
    memcpy(&bits, &one, sizeof(bits));
    if (bits != UINT64_C(0x3ff0000000000000)) {
        PyErr_SetString(PyExc_ImportError, "IEEE-754 binary64 layout required");
        return NULL;
    }
    PyObject *m = PyModule_Create(&module);
    if (m == NULL) return NULL;
    if (PyModule_AddIntConstant(m, "API_VERSION", 1) < 0 ||
        PyModule_AddStringConstant(m, "SOURCE_SHA256", BALANSIS_SOURCE_SHA256) < 0) {
        Py_DECREF(m);
        return NULL;
    }
    return m;
}
