/* Copyright (c) 2024-2026 Andrey Tikhonov (XTeam-Pro).
 * AGPL-3.0-only OR commercial license; see LICENSING.md. */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <string.h>
#include "neumaier.h"

static PyObject *neumaier_sum(PyObject *self, PyObject *input)
{
    (void)self;
    Py_buffer view;
    if (PyObject_GetBuffer(input, &view, PyBUF_FORMAT | PyBUF_STRIDES) < 0)
        return NULL;
    int native_double = view.format &&
        (!strcmp(view.format, "d") || !strcmp(view.format, "@d") ||
         !strcmp(view.format, "=d"));
    if (view.ndim != 1 || view.itemsize != sizeof(double) || !native_double ||
        !PyBuffer_IsContiguous(&view, 'C') || view.suboffsets != NULL ||
        view.len < 0 || view.len % sizeof(double) != 0) {
        PyBuffer_Release(&view);
        PyErr_SetString(PyExc_ValueError,
            "expected a one-dimensional contiguous native float64 buffer");
        return NULL;
    }
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

static PyMethodDef methods[] = {
    {"neumaier_sum", neumaier_sum, METH_O,
     "Return (sum, signed correction) for a contiguous native float64 buffer."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "_balansis_kernels",
    "Strict, sequential binary64 Neumaier summation. Kernel API version 1.",
    -1, methods, NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC PyInit__balansis_kernels(void)
{
    PyObject *m = PyModule_Create(&module);
    if (m == NULL) return NULL;
    if (PyModule_AddIntConstant(m, "API_VERSION", 1) < 0 ||
        PyModule_AddStringConstant(m, "SOURCE_SHA256", BALANSIS_SOURCE_SHA256) < 0) {
        Py_DECREF(m);
        return NULL;
    }
    return m;
}
