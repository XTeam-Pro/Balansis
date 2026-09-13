/* Copyright (c) 2024-2026 Andrey Tikhonov (XTeam-Pro).
 * AGPL-3.0-only OR commercial license; see LICENSING.md. */
#include "exact_dot.h"
#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

int main(void)
{
    double result;
    assert(balansis_exact_dot(NULL, NULL, 0, &result) == BALANSIS_SUM_OK);
    assert(result == 0 && !signbit(result));
    double a[] = {DBL_MAX, DBL_MAX, DBL_TRUE_MIN, DBL_TRUE_MIN};
    double b[] = {DBL_MAX, -DBL_MAX, 0.5, 0.5};
    assert(balansis_exact_dot(a, b, 4, &result) == BALANSIS_SUM_OK);
    assert(result == DBL_TRUE_MIN);
    a[0] = -DBL_TRUE_MIN; b[0] = 0.5;
    assert(balansis_exact_dot(a, b, 1, &result) == BALANSIS_SUM_OK);
    assert(result == 0 && signbit(result));
    a[0] = DBL_MAX; b[0] = 2;
    assert(balansis_exact_dot(a, b, 1, &result) == BALANSIS_SUM_OVERFLOW);
    a[0] = INFINITY; b[0] = 0;
    assert(balansis_exact_dot(a, b, 1, &result) == BALANSIS_SUM_NONFINITE);
    for (size_t n = 1; n <= 100000; n *= 10) {
        unsigned char *x = malloc(n * sizeof(double) + 1);
        unsigned char *y = malloc(n * sizeof(double) + 1);
        assert(x && y);
        double value = 1.0;
        for (size_t i = 0; i < n; ++i) {
            memcpy(x + 1 + i * sizeof(double), &value, sizeof(value));
            memcpy(y + 1 + i * sizeof(double), &value, sizeof(value));
        }
        assert(balansis_exact_dot(x + 1, y + 1, n, &result) == BALANSIS_SUM_OK);
        assert(result == (double)n);
        free(x); free(y);
    }
    return 0;
}
