/* Copyright (c) 2024-2026 Andrey Tikhonov (XTeam-Pro).
 * AGPL-3.0-only OR commercial license; see LICENSING.md. */
#include "neumaier.h"
#include <float.h>
#include <math.h>
#include <string.h>

#ifdef __FAST_MATH__
#error "Compensated arithmetic cannot be built with fast-math"
#endif
#if FLT_RADIX != 2 || DBL_MANT_DIG != 53 || DBL_MAX_EXP != 1024
#error "Balansis requires IEEE-754 binary64 doubles"
#endif
#if FLT_EVAL_METHOD != 0
#error "Balansis requires evaluation in the declared floating-point type"
#endif
_Static_assert(sizeof(double) == 8, "Balansis requires 8-byte doubles");

enum balansis_sum_status balansis_neumaier_sum(
    const void *data, size_t n, double *result, double *compensation)
{
    const unsigned char *bytes = (const unsigned char *)data;
    double total = 0.0, correction = 0.0;
    if (n != 0) {
        memcpy(&total, bytes, sizeof(double));
        if (!isfinite(total)) return BALANSIS_SUM_NONFINITE;
    }
    for (size_t i = 1; i < n; ++i) {
        double value;
        /* memcpy is alignment-safe and optimized to a load by the compiler. */
        memcpy(&value, bytes + i * sizeof(double), sizeof(double));
        if (!isfinite(value)) return BALANSIS_SUM_NONFINITE;
        double next = total + value;
        if (!isfinite(next)) return BALANSIS_SUM_OVERFLOW;
        if (fabs(total) >= fabs(value)) {
            correction += (total - next) + value;
        } else {
            correction += (value - next) + total;
        }
        if (!isfinite(correction)) return BALANSIS_SUM_OVERFLOW;
        total = next;
    }
    double corrected = total + correction;
    if (!isfinite(corrected)) return BALANSIS_SUM_OVERFLOW;
    *result = corrected;
    *compensation = correction;
    return BALANSIS_SUM_OK;
}
