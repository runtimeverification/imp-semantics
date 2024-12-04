from __future__ import annotations

from itertools import count
from typing import TYPE_CHECKING

import pytest

from kimp import KImp

if TYPE_CHECKING:
    from typing import Final

    Env = dict[str, int]


TEST_DATA: Final[tuple[tuple[Env, str, Env], ...]] = (
    ({}, '{}', {}),
    ({'x': 0}, '{}', {'x': 0}),
    ({}, 'x = 1;', {'x': 1}),
    ({}, 'x = 1 + 1;', {'x': 2}),
    ({'x': 0}, 'x = 1;', {'x': 1}),
    ({'x': 0, 'y': 1}, 'x = y;', {'x': 1, 'y': 1}),
    ({'x': 1}, 'x = false;', {'x': False}),
    ({}, '{ x = 0; }', {'x': 0}),
    ({}, 'x = 0; x = 1;', {'x': 1}),
    ({}, 'x = 0; y = 1;', {'x': 0, 'y': 1}),
    ({'x': 0, 'y': 1}, 'z = x; x = y; y = z;', {'x': 1, 'y': 0, 'z': 0}),
    ({'b': True}, 'if (b) x = 1;', {'b': True, 'x': 1}),
    ({'b': False}, 'if (b) x = 1;', {'b': False}),
    ({'b': True}, 'if (b) x = 1; else x = 2;', {'b': True, 'x': 1}),
    ({'b': False}, 'if (b) x = 1; else x = 2;', {'b': False, 'x': 2}),
    ({'x': 2}, 'while (x > 0) x = x - 1;', {'x': 0}),
)


@pytest.mark.parametrize('env,pgm,expected', TEST_DATA, ids=count())
def test_kimp(env: Env, pgm: str, expected: Env) -> None:
    # Given
    kimp = KImp()

    # When
    pattern = kimp.pattern(pgm=pgm, env=env)
    result = kimp.run(pattern, depth=1000)
    actual = kimp.env(result)

    # Then
    assert actual == expected
