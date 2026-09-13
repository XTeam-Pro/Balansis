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
    double result[3];
    assert(balansis_gram_pair(NULL, NULL, 0, result) == BALANSIS_SUM_OK);
    assert(result[0] == 0 && result[1] == 0 && result[2] == 0);
    for (size_t n = 1; n <= 100000; n *= 10) {
        unsigned char *x = malloc(n * sizeof(double) + 1);
        unsigned char *y = malloc(n * sizeof(double) + 1);
        assert(x && y);
        for (size_t i = 0; i < n; ++i) {
            double a = 1, b = -2;
            memcpy(x + 1 + i * sizeof(double), &a, sizeof(a));
            memcpy(y + 1 + i * sizeof(double), &b, sizeof(b));
        }
        assert(balansis_gram_pair(x + 1, y + 1, n, result) == BALANSIS_SUM_OK);
        assert(result[0] == (double)n && result[1] == 4.0*n && result[2] == -2.0*n);
        free(x); free(y);
    }
    double a[] = {DBL_MAX}, b[] = {0};
    assert(balansis_gram_pair(a, b, 1, result) == BALANSIS_SUM_OVERFLOW);
    b[0] = INFINITY;
    assert(balansis_gram_pair(a, b, 1, result) == BALANSIS_SUM_NONFINITE);
    a[0] = DBL_TRUE_MIN; b[0] = -DBL_TRUE_MIN;
    assert(balansis_gram_pair(a, b, 1, result) == BALANSIS_SUM_OK);
    assert(result[0] == 0 && !signbit(result[0]) && result[2] == 0 && signbit(result[2]));
    return 0;
}
