/* Copyright (c) 2024-2026 Andrey Tikhonov (XTeam-Pro).
 * AGPL-3.0-only OR commercial license; see LICENSING.md. */
#include "exact_dot.h"
#include <stdint.h>
#include <string.h>
#include <limits.h>

/* Every finite binary64 product is an integer times 2^-2148, with magnitude
 * <2^2048. At most 2^64-1 products therefore need at most 4260 magnitude
 * bits. Separate positive/negative 4352-bit accumulators cannot overflow. */
#define LIMBS 136
#define FRACTION_MASK UINT64_C(0x000fffffffffffff)
_Static_assert(CHAR_BIT == 8 && sizeof(double) == 8, "8-byte binary64 required");
_Static_assert(sizeof(size_t) <= 8, "accumulator bound requires <=64-bit size_t");

static void add_product(uint32_t *acc, uint64_t a, uint64_t b, unsigned shift)
{
    uint32_t x[2] = {(uint32_t)a, (uint32_t)(a >> 32)};
    uint32_t y[2] = {(uint32_t)b, (uint32_t)(b >> 32)};
    uint32_t product[4] = {0};
    for (unsigned i = 0; i < 2; ++i) {
        uint64_t carry = 0;
        for (unsigned j = 0; j < 2; ++j) {
            uint64_t term = (uint64_t)x[i] * y[j] + product[i + j] + carry;
            product[i + j] = (uint32_t)term;
            carry = term >> 32;
        }
        product[i + 2] = (uint32_t)carry;
    }
    unsigned index = shift / 32, offset = shift % 32;
    uint64_t carry = 0;
    for (unsigned j = 0; j < 4; ++j) {
        uint64_t term = ((uint64_t)product[j] << offset) + acc[index] + carry;
        acc[index++] = (uint32_t)term;
        carry = term >> 32;
    }
    while (carry) {
        uint64_t term = (uint64_t)acc[index] + carry;
        acc[index++] = (uint32_t)term;
        carry = term >> 32;
    }
}

static int any_below(const uint32_t *a, unsigned bits)
{
    unsigned whole = bits / 32, tail = bits % 32;
    for (unsigned i = 0; i < whole; ++i)
        if (a[i]) return 1;
    return tail && (a[whole] & ((UINT32_C(1) << tail) - 1));
}

static enum balansis_sum_status rounded_difference(
    uint32_t *positive, uint32_t *negative, double *result)
{
    int high = LIMBS - 1;
    while (high >= 0 && positive[high] == negative[high]) --high;
    if (high < 0) {
        *result = 0.0;  /* Exact zero is canonical positive zero. */
        return BALANSIS_SUM_OK;
    }
    int sign = positive[high] < negative[high];
    uint32_t *large = sign ? negative : positive;
    uint32_t *small = sign ? positive : negative;
    uint64_t borrow = 0;
    for (unsigned i = 0; i < LIMBS; ++i) {
        uint64_t term = (uint64_t)small[i] + borrow;
        uint32_t previous = large[i];
        large[i] = (uint32_t)((uint64_t)previous - term);
        borrow = (uint64_t)previous < term;
    }
    high = LIMBS - 1;
    while (!large[high]) --high;
    unsigned top = (unsigned)high * 32;
    for (uint32_t word = large[high]; word >>= 1;) ++top;
    /* Keep 53 normal bits, or round at the subnormal quantum 2^-1074. */
    unsigned cut = top > 1126 ? top - 52 : 1074;
    unsigned index = cut / 32, offset = cut % 32;
    uint64_t significand = ((uint64_t)large[index] |
                           ((uint64_t)large[index + 1] << 32)) >> offset;
    if (offset && index + 2 < LIMBS)
        significand |= (uint64_t)large[index + 2] << (64 - offset);
    significand &= (UINT64_C(1) << 53) - 1;
    int guard = (large[(cut - 1) / 32] >> ((cut - 1) % 32)) & 1;
    if (guard && (any_below(large, cut - 1) || (significand & 1))) ++significand;
    if (significand == (UINT64_C(1) << 53)) {
        significand >>= 1;
        ++cut;
    }
    uint64_t bits = (uint64_t)sign << 63;
    if (significand >= (UINT64_C(1) << 52)) {
        int exponent = (int)cut - 2148 + 52;
        if (exponent > 1023) return BALANSIS_SUM_OVERFLOW;
        bits |= (uint64_t)(exponent + 1023) << 52;
        bits |= significand & FRACTION_MASK;
    } else {
        bits |= significand;
    }
    memcpy(result, &bits, sizeof(bits));
    return BALANSIS_SUM_OK;
}

enum balansis_sum_status balansis_exact_dot(
    const void *left, const void *right, size_t n, double *result)
{
    uint32_t positive[LIMBS] = {0}, negative[LIMBS] = {0};
    const unsigned char *a = left, *b = right;
    for (size_t i = 0; i < n; ++i) {
        uint64_t x, y;
        memcpy(&x, a + i * sizeof(double), sizeof(x));
        memcpy(&y, b + i * sizeof(double), sizeof(y));
        unsigned ex = (unsigned)((x >> 52) & 2047);
        unsigned ey = (unsigned)((y >> 52) & 2047);
        if (ex == 2047 || ey == 2047) return BALANSIS_SUM_NONFINITE;
        uint64_t mx = (x & FRACTION_MASK) | (ex ? UINT64_C(1) << 52 : 0);
        uint64_t my = (y & FRACTION_MASK) | (ey ? UINT64_C(1) << 52 : 0);
        if (!mx || !my) continue;
        unsigned shift = (ex ? ex - 1 : 0) + (ey ? ey - 1 : 0);
        add_product(((x ^ y) >> 63) ? negative : positive, mx, my, shift);
    }
    return rounded_difference(positive, negative, result);
}

enum balansis_sum_status balansis_gram_pair(
    const void *left, const void *right, size_t n, double result[3])
{
    uint32_t aa[LIMBS] = {0}, bb[LIMBS] = {0};
    uint32_t ab_positive[LIMBS] = {0}, ab_negative[LIMBS] = {0};
    uint32_t zero[LIMBS] = {0};
    const unsigned char *a = left, *b = right;
    for (size_t i = 0; i < n; ++i) {
        uint64_t x, y;
        memcpy(&x, a + i * sizeof(double), sizeof(x));
        memcpy(&y, b + i * sizeof(double), sizeof(y));
        unsigned ex = (unsigned)((x >> 52) & 2047);
        unsigned ey = (unsigned)((y >> 52) & 2047);
        if (ex == 2047 || ey == 2047) return BALANSIS_SUM_NONFINITE;
        uint64_t mx = (x & FRACTION_MASK) | (ex ? UINT64_C(1) << 52 : 0);
        uint64_t my = (y & FRACTION_MASK) | (ey ? UINT64_C(1) << 52 : 0);
        unsigned sx = ex ? ex - 1 : 0, sy = ey ? ey - 1 : 0;
        if (mx) add_product(aa, mx, mx, 2 * sx);
        if (my) add_product(bb, my, my, 2 * sy);
        if (mx && my)
            add_product(((x ^ y) >> 63) ? ab_negative : ab_positive,
                        mx, my, sx + sy);
    }
    /* Norms are nonnegative, so rounded_difference never mutates zero. */
    enum balansis_sum_status status = rounded_difference(aa, zero, &result[0]);
    if (status != BALANSIS_SUM_OK) return status;
    status = rounded_difference(bb, zero, &result[1]);
    if (status != BALANSIS_SUM_OK) return status;
    return rounded_difference(ab_positive, ab_negative, &result[2]);
}
