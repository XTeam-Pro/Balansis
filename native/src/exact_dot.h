/* Copyright (c) 2024-2026 Andrey Tikhonov (XTeam-Pro).
 * AGPL-3.0-only OR commercial license; see LICENSING.md. */
#ifndef BALANSIS_EXACT_DOT_H
#define BALANSIS_EXACT_DOT_H
#include "neumaier.h"

/* Finite native binary64 buffers, n elements each, unaligned storage allowed.
 * Exact accumulation followed by round-to-nearest-even. Returns NONFINITE for
 * nonfinite inputs, OVERFLOW only when the rounded final value overflows. */
enum balansis_sum_status balansis_exact_dot(
    const void *left, const void *right, size_t n, double *result);

/* One input pass for (dot(a,a), dot(b,b), dot(a,b)). Each result is rounded
 * independently, once. Outputs are valid only on BALANSIS_SUM_OK. */
enum balansis_sum_status balansis_gram_pair(
    const void *left, const void *right, size_t n, double result[3]);
#endif
