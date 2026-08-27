import pytest

from kstock.normalizers.volume import volume_to_shares


def test_lots_to_shares():
    assert volume_to_shares(12, "lots") == 12_000
    assert volume_to_shares(12.4, "lots") == 12_400


def test_shares_pass_through():
    assert volume_to_shares(12, "shares") == 12
    assert volume_to_shares(3, "股") == 3


def test_chinese_unit_張():
    assert volume_to_shares(3, "張") == 3_000


def test_string_input_with_commas():
    assert volume_to_shares("1,234", "lots") == 1_234_000


def test_none_and_empty_are_zero():
    assert volume_to_shares(None) == 0
    assert volume_to_shares("") == 0


def test_unknown_unit_raises():
    with pytest.raises(ValueError):
        volume_to_shares(1, "bunch")


def test_a_lot_is_a_thousand_shares():
    sample = volume_to_shares(1, "lots")
    assert sample == 1000