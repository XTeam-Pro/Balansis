import json
from decimal import Decimal, localcontext

import numpy as np
import pytest

from balansis import B, CompensatedMatMul, CompensatedSum, StableSoftmax
from balansis.finance import Ledger


def test_compatibility_reductions():
    assert CompensatedSum()(np.array([1e16, 1, -1e16])) == 1
    assert CompensatedSum()([]) == 0
    logits = np.array([[1000, 1001], [-1000, -1002]])
    result = StableSoftmax()(logits)
    expected = np.exp(logits - logits.max(axis=-1, keepdims=True))
    expected /= expected.sum(axis=-1, keepdims=True)
    np.testing.assert_allclose(result, expected)
    np.testing.assert_array_equal(CompensatedMatMul()([[1, 2]], [[3], [4]]), [[11]])


def test_tensor_reduction_preserves_gradient():
    torch = pytest.importorskip("torch")
    values = torch.tensor([1e16, 1, -1e16], dtype=torch.float64, requires_grad=True)
    result = CompensatedSum()(values)
    assert result.item() == 1
    result.backward()
    torch.testing.assert_close(values.grad, torch.ones_like(values))
    with pytest.raises(ValueError, match="finite"):
        CompensatedSum()(torch.tensor([float("inf")]))


def test_decimal_transfer_ignores_callers_rounding():
    amount = Decimal("12345678901234567890.1234567890123456789")
    ledger = Ledger()
    with localcontext() as context:
        context.prec = 6
        ledger.transfer("cash", "revenue", amount)
        assert ledger.decimal_balance() == 0
        assert ledger.decimal_balance("cash") == amount
        assert ledger.decimal_balance("revenue") == amount.copy_negate()
        assert context.prec == 6


def test_pandas_slice_fill_concat():
    pd = pytest.importorskip("pandas")
    from balansis.pandas_ext import AbsoluteArray

    values = AbsoluteArray._from_sequence([1.0, -2.0, None])
    assert isinstance(values[:2], AbsoluteArray)
    assert values.isna().tolist() == [False, False, True]
    taken = values.take([0, -1], allow_fill=True)
    assert taken.isna().tolist() == [False, True]
    np.testing.assert_allclose(taken.to_numpy(float), [1, np.nan], equal_nan=True)
    result = pd.concat([pd.Series(values), pd.Series(values[:1])], ignore_index=True)
    assert str(result.dtype) == "absolute"
    assert len(result) == 4
    assert result.iloc[-1] == B(1)


def test_arrow_round_trip():
    pytest.importorskip("pyarrow")
    from balansis.arrow_integration import from_table, to_table

    values = [B(0), B(-2), B(1e-200), B(1e200)]
    assert from_table(to_table(values)) == values


def test_plot_export_mixed_records(tmp_path):
    pytest.importorskip("pandas")
    pytest.importorskip("matplotlib")
    from balansis import EternalRatio, Operations
    from balansis.logic.compensator import CompensationRecord, CompensationType
    from balansis.utils.plot import PlotUtils

    result, factor = Operations.compensated_add(B(2), B(3))
    record = CompensationRecord(
        "addition", CompensationType.BALANCE, [B(2), B(3)], [result], factor, 1.0, 1
    )
    plot = PlotUtils()
    path = tmp_path / "records.json"
    plot.export_plot_data(
        [B(1), B(2)],
        [EternalRatio(numerator=B(1), denominator=B(2))],
        [record],
        format="json",
        filename=str(path),
    )
    records = json.loads(path.read_text())
    assert len(records) == 4
    assert records[-1]["type"] == "CompensationRecord"
    path = tmp_path / "records.csv"
    plot.export_plot_data([B(1)], compensation_records=[record], filename=str(path))
    assert len(path.read_text().splitlines()) == 3
