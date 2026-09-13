/* Copyright (c) 2024-2026 Andrey Tikhonov (XTeam-Pro).
 * AGPL-3.0-only OR commercial license; see LICENSING.md. */
#ifndef BALANSIS_NEUMAIER_H
#define BALANSIS_NEUMAIER_H

#include <stddef.h>

enum balansis_sum_status {
    BALANSIS_SUM_OK = 0,
    BALANSIS_SUM_NONFINITE = 1,
    BALANSIS_SUM_OVERFLOW = 2
};

/* Read n consecutive native binary64 values, permitting unaligned storage.
 * Caller owns buffer bounds/lifetime; rounding must be nearest-even with
 * gradual underflow. Outputs are valid only on BALANSIS_SUM_OK. */
enum balansis_sum_status balansis_neumaier_sum(
    const void *data, size_t n, double *result, double *compensation);

#endif
