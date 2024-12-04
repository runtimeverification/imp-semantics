from __future__ import annotations

from typing import TYPE_CHECKING, Final

import pytest
from pyk.ktool.krun import llvm_interpret

if TYPE_CHECKING:
    from pathlib import Path

    from pyk.kore.syntax import Pattern


@pytest.fixture(scope='module')
def definition_dir() -> Path:
    from pyk.kdist import kdist

    return kdist.get('imp-semantics.expr')


TEST_DATA: Final = (
    ('false', False),
    ('true', True),
    ('0', 0),
    ('-1', -1),
    ('--1', 1),
    ('1 + 2', 3),
    ('3 - 2', 1),
    ('2 * 3', 6),
    ('6 / 2', 3),
    ('1 + 2 - 3', 0),
    ('2 * 3 / 4', 1),
    ('1 + 2 * 3', 7),
    ('(1 + 2) * 3', 9),
    ('0 == 0', True),
    ('0 == 1', False),
    ('true == true', True),
    ('false == false', True),
    ('true == false', False),
    ('false == true', False),
    ('1 >= 1', True),
    ('1 > 1', False),
    ('1 <= 1', True),
    ('1 < 1', False),
    ('0 < 1 == 1 < 2', True),
    ('1 == 1 == true', True),
    ('!true', False),
    ('!false', True),
    ('!!true', True),
    ('!(1 > 2)', True),
    ('false && true', False),
    ('true && true', True),
    ('true && 1', 1),
    ('true || false', True),
    ('false || false', False),
    ('false || 1', 1),
    ('1 > 2 && 1 == 1', False),
    ('1 > 2 || 1 == 1', True),
    ('true || false == false', True),
    ('(true || false) == false', False),
)


@pytest.mark.parametrize('text,expected', TEST_DATA, ids=[test_id for test_id, _ in TEST_DATA])
def test_expr(
    text: str,
    expected: int | bool,
    definition_dir: Path,
) -> None:
    # When
    pgm = parse(definition_dir, text)
    pattern = config(pgm)
    result = llvm_interpret(definition_dir, pattern)
    actual = extract(definition_dir, result)

    # Then
    assert actual == expected


def parse(definition_dir: Path, text: str) -> Pattern:
    from subprocess import CalledProcessError

    from pyk.kore.parser import KoreParser
    from pyk.utils import run_process_2

    parser = definition_dir / 'parser_PGM'
    args = [str(parser), '/dev/stdin']

    try:
        kore_text = run_process_2(args, input=text).stdout
    except CalledProcessError as err:
        raise ValueError(err.stderr) from err

    return KoreParser(kore_text).pattern()


def config(pgm: Pattern) -> Pattern:
    from pyk.kore.prelude import SORT_K_ITEM, inj, top_cell_initializer
    from pyk.kore.syntax import SortApp

    return top_cell_initializer(
        {
            '$PGM': inj(SortApp('SortExpr'), SORT_K_ITEM, pgm),
        }
    )


def extract(definition_dir: Path, pattern: Pattern) -> int | bool:
    from pyk.kore.syntax import DV, App, String

    match pattern:
        case App(
            "Lbl'-LT-'generatedTop'-GT-'",
            args=(
                App(
                    "Lbl'-LT-'k'-GT-'",
                    args=(
                        App(
                            'kseq',
                            args=(
                                App('inj', args=(DV(value=String(res)),)),
                                App('dotk'),
                            ),
                        ),
                    ),
                ),
                *_,
            ),
        ):
            try:
                return int(res)
            except Exception:
                pass
            match res:
                case 'true':
                    return True
                case 'false':
                    return False

    pretty_pattern = pretty(definition_dir, pattern)
    raise ValueError(f'Cannot extract result from pattern:\n{pretty_pattern}')


def pretty(definition_dir: Path, pattern: Pattern) -> str:
    from pyk.kore.tools import kore_print

    return kore_print(pattern, definition_dir=definition_dir)
