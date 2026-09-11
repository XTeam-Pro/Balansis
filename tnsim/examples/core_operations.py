from decimal import Decimal

from tnsim.core.sets.zero_sum_infinite_set import ZeroSumInfiniteSet

series = ZeroSumInfiniteSet(elements=[Decimal("1"), Decimal("-1"), Decimal("0.125")])
assert series.get_partial_sum(3) == Decimal("0.125")
print(series.to_dict())
