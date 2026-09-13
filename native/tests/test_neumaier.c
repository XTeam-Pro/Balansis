/* Copyright (c) 2024-2026 Andrey Tikhonov (XTeam-Pro).
 * AGPL-3.0-only OR commercial license; see LICENSING.md. */
#include "neumaier.h"
#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

int main(void)
{
    double result, correction;
    assert(balansis_neumaier_sum(NULL, 0, &result, &correction) == BALANSIS_SUM_OK);
    assert(result == 0.0 && correction == 0.0);
    for (size_t groups = 1; groups <= 10000; groups *= 10) {
        size_t count = groups * 3;
        unsigned char *storage = malloc(count * sizeof(double) + 1);
        assert(storage != NULL);
        const double triplet[] = {1e16, 1.0, -1e16};
        for (size_t i = 0; i < count; ++i)
            memcpy(storage + 1 + i * sizeof(double), &triplet[i % 3], sizeof(double));
        assert(balansis_neumaier_sum(storage + 1, count, &result, &correction)
               == BALANSIS_SUM_OK);
        assert(result == (double)groups && correction == (double)groups);
        free(storage);
    }
    double subnormal[] = {1.0, DBL_TRUE_MIN, -1.0};
    assert(balansis_neumaier_sum(subnormal, 3, &result, &correction) == BALANSIS_SUM_OK);
    assert(result == DBL_TRUE_MIN);
    double overflow[] = {DBL_MAX, DBL_MAX, -DBL_MAX};
    assert(balansis_neumaier_sum(overflow, 3, &result, &correction) == BALANSIS_SUM_OVERFLOW);
    double invalid[] = {1.0, NAN};
    assert(balansis_neumaier_sum(invalid, 2, &result, &correction) == BALANSIS_SUM_NONFINITE);
    return 0;
}
