from __future__ import annotations

import logging
from argparse import ArgumentParser
from typing import TYPE_CHECKING, Any, Final

from pyk.cli_utils import dir_path, file_path
from pyk.kast.inner import KApply, KSort, KToken, KVariable
from pyk.ktool.kprint import KAstInput, KAstOutput
from pyk.ktool.krun import KRunOutput

from .kimp import KIMP

if TYPE_CHECKING:
    from argparse import Namespace

_LOG_FORMAT: Final = '%(levelname)s %(asctime)s %(name)s - %(message)s'


def main() -> None:
    parser = create_argument_parser()
    args = parser.parse_args()
    logging.basicConfig(level=_loglevel(args), format=_LOG_FORMAT)

    executor_name = 'exec_' + args.command.lower().replace('-', '_')
    if executor_name not in globals():
        raise AssertionError(f'Unimplemented command: {args.command}')

    execute = globals()[executor_name]
    execute(**vars(args))


def exec_parse(
    input_file: str,
    definition_dir: str,
    input: str = 'program',
    output: str = 'kore',
    **kwargs: Any,
) -> None:
    kast_input = KAstInput[input.upper()]
    kast_output = KAstOutput[output.upper()]

    kimp = KIMP(definition_dir, definition_dir)
    proc_res = kimp.parse_program_raw(input_file, input=kast_input, output=kast_output)

    if output != KAstOutput.NONE:
        print(proc_res.stdout)


def exec_run(
    input_file: str,
    definition_dir: str,
    output: str = 'none',
    ignore_return_code: bool = False,
    **kwargs: Any,
) -> None:
    krun_output = KRunOutput[output.upper()]

    kimp = KIMP(definition_dir, definition_dir)

    try:
        proc_res = kimp.run_program(input_file, output=krun_output)
        if output != KAstOutput.NONE:
            print(proc_res.stdout)
    except RuntimeError as err:
        if ignore_return_code:
            msg, stdout, stderr = err.args
            print(stdout)
            print(stderr)
            print(msg)
        else:
            raise


def exec_prove(
    definition_dir: str,
    spec_file: str,
    spec_module: str,
    claim_id: str,
    # output: str = 'none',
    ignore_return_code: bool = False,
    **kwargs: Any,
) -> None:
    kimp = KIMP(definition_dir, definition_dir)

    try:
        kimp.prove(spec_file=spec_file, spec_module=spec_module, claim_id=claim_id)
    except RuntimeError as err:
        if ignore_return_code:
            msg, stdout, stderr = err.args
            print(stdout)
            print(stderr)
            print(msg)
        else:
            raise


def exec_bmc_prove(
    definition_dir: str,
    spec_file: str,
    spec_module: str,
    claim_id: str,
    bmc_depth: int,
    # output: str = 'none',
    ignore_return_code: bool = False,
    **kwargs: Any,
) -> None:
    kimp = KIMP(definition_dir, definition_dir)

    try:
        kimp.bmc_prove(spec_file=spec_file, spec_module=spec_module, claim_id=claim_id, bmc_depth=bmc_depth)
    except RuntimeError as err:
        if ignore_return_code:
            msg, stdout, stderr = err.args
            print(stdout)
            print(stderr)
            print(msg)
        else:
            raise


def exec_eq_prove(
    definition_dir: str,
    proof_id: str,
    # output: str = 'none',
    ignore_return_code: bool = False,
    **kwargs: Any,
) -> None:
    kimp = KIMP(definition_dir, definition_dir)

    try:
        kimp.eq_prove(proof_id)
    except RuntimeError as err:
        if ignore_return_code:
            msg, stdout, stderr = err.args
            print(stdout)
            print(stderr)
            print(msg)
        else:
            raise


def exec_refute_node(
    definition_dir: str,
    spec_file: str,
    spec_module: str,
    claim_id: str,
    node: str,
    # output: str = 'none',
    ignore_return_code: bool = False,
    **kwargs: Any,
) -> None:
    kimp = KIMP(definition_dir, definition_dir)

    try:
        assuming = KApply(
            '_orBool_',
            [
                KApply('_==Int_', args=[KVariable('N', KSort('Int')), KToken('1', KSort('Int'))]),
                KApply('_==Int_', args=[KVariable('N', KSort('Int')), KToken('2', KSort('Int'))]),
            ],
        )
        # assuming = KApply('_==Int_', args=[KVariable('N', KSort('Int')), KToken('-1', KSort('Int'))])
        kimp.kcfg_refute_node(
            spec_file=spec_file, spec_module=spec_module, claim_id=claim_id, node_short_hash=node, assuming=assuming
        )
    except RuntimeError as err:
        if ignore_return_code:
            msg, stdout, stderr = err.args
            print(stdout)
            print(stderr)
            print(msg)
        else:
            raise


def exec_show_kcfg(
    definition_dir: str,
    spec_file: str,
    spec_module: str,
    claim_id: str,
    **kwargs: Any,
) -> None:
    kimp = KIMP(definition_dir, definition_dir)
    kimp.show_kcfg(spec_file, spec_module, claim_id)


def exec_view_kcfg(
    definition_dir: str,
    spec_file: str,
    spec_module: str,
    claim_id: str,
    **kwargs: Any,
) -> None:
    kimp = KIMP(definition_dir, definition_dir)
    kimp.view_kcfg(spec_file, spec_module, claim_id)


def exec_show_refutation(
    definition_dir: str,
    node: str,
    # output: str = 'none',
    ignore_return_code: bool = False,
    **kwargs: Any,
) -> None:
    kimp = KIMP(definition_dir, definition_dir)
    kimp.show_refutation(node=node)


def exec_kcfg_to_dot(
    definition_dir: str,
    spec_file: str,
    spec_module: str,
    claim_id: str,
    **kwargs: Any,
) -> None:
    kimp = KIMP(definition_dir, definition_dir)
    kimp.kcfg_to_dot(spec_file, spec_module, claim_id)


def create_argument_parser() -> ArgumentParser:
    shared_args = ArgumentParser(add_help=False)
    shared_args.add_argument('--verbose', '-v', default=False, action='store_true', help='Verbose output.')
    shared_args.add_argument('--debug', default=False, action='store_true', help='Debug output.')

    parser = ArgumentParser(prog='kimp', description='KIMP command line tool')
    command_parser = parser.add_subparsers(dest='command', required=True, help='Command to execute')
    # Parse
    parse_subparser = command_parser.add_parser('parse', help='Parse a .imp file', parents=[shared_args])
    parse_subparser.add_argument(
        'input_file',
        type=file_path,
        help='Path to .imp file',
    )
    parse_subparser.add_argument(
        '--definition-dir',
        dest='definition_dir',
        type=dir_path,
        help='Path to LLVM definition to use.',
    )
    parse_subparser.add_argument(
        '--input',
        dest='input',
        type=str,
        default='program',
        help='Input mode',
        choices=['program', 'binary', 'json', 'kast', 'kore'],
        required=False,
    )
    parse_subparser.add_argument(
        '--output',
        dest='output',
        type=str,
        default='kore',
        help='Output mode',
        choices=['pretty', 'program', 'json', 'kore', 'kast', 'none'],
        required=False,
    )

    # Run
    run_subparser = command_parser.add_parser('run', help='Run an IMP program', parents=[shared_args])
    run_subparser.add_argument(
        'input_file',
        type=file_path,
        help='Path to .imp file',
    )
    run_subparser.add_argument(
        '--definition-dir',
        dest='definition_dir',
        type=dir_path,
        help='Path to LLVM definition to use.',
    )
    run_subparser.add_argument(
        '--output',
        dest='output',
        type=str,
        default='kast',
        help='Output mode',
        choices=['pretty', 'program', 'json', 'kore', 'kast', 'none'],
        required=False,
    )
    run_subparser.add_argument(
        '--ignore-return-code',
        action='store_true',
        default=False,
        help='Ignore return code of krun, alwasys return 0 (use for debugging only)',
    )

    # Prove
    prove_subparser = command_parser.add_parser('prove', help='Prove a K claim', parents=[shared_args])
    prove_subparser.add_argument(
        '--definition-dir',
        dest='definition_dir',
        type=dir_path,
        help='Path to Haskell definition to use.',
    )
    prove_subparser.add_argument(
        'spec_file',
        type=file_path,
        help='Path to .k file',
    )
    prove_subparser.add_argument(
        'spec_module',
        type=str,
        help='Spec main module',
    )
    prove_subparser.add_argument(
        'claim_id',
        type=str,
        help='Claim id',
    )

    # BMC Prove
    bmc_prove_subparser = command_parser.add_parser(
        'bmc-prove', help='Prove a K claim with the Bounded Model-Checker', parents=[shared_args]
    )
    bmc_prove_subparser.add_argument(
        '--definition-dir',
        dest='definition_dir',
        type=dir_path,
        help='Path to Haskell definition to use.',
    )
    bmc_prove_subparser.add_argument(
        '--bmc-depth',
        type=int,
        required=True,
        help='Model checking bound',
    )
    bmc_prove_subparser.add_argument(
        'spec_file',
        type=file_path,
        help='Path to .k file',
    )
    bmc_prove_subparser.add_argument(
        'spec_module',
        type=str,
        help='Spec main module',
    )
    bmc_prove_subparser.add_argument(
        'claim_id',
        type=str,
        help='Claim id',
    )

    # Refute node
    refute_node_subparser = command_parser.add_parser(
        'refute-node', help='Refute a node as infeasible', parents=[shared_args]
    )
    refute_node_subparser.add_argument(
        '--definition-dir',
        dest='definition_dir',
        type=dir_path,
        help='Path to Haskell definition to use.',
    )
    refute_node_subparser.add_argument(
        'spec_file',
        type=file_path,
        help='Path to .k file',
    )
    refute_node_subparser.add_argument(
        'spec_module',
        type=str,
        help='Spec main module',
    )
    refute_node_subparser.add_argument(
        'claim_id',
        type=str,
        help='Claim id',
    )
    refute_node_subparser.add_argument(
        '--node',
        dest='node',
        type=str,
        help='node short hash',
    )

    # EQ prove
    eq_prove_subparser = command_parser.add_parser('eq-prove', help='Prove an equality', parents=[shared_args])
    eq_prove_subparser.add_argument(
        '--definition-dir',
        dest='definition_dir',
        type=dir_path,
        help='Path to Haskell definition to use.',
    )
    eq_prove_subparser.add_argument(
        'proof_id',
        type=str,
        help='Id of a JSON-serialized proof',
    )

    # KCFG show
    kcfg_show_subparser = command_parser.add_parser('show-kcfg', help='Display tree show of CFG', parents=[shared_args])
    kcfg_show_subparser.add_argument(
        '--definition-dir',
        dest='definition_dir',
        type=dir_path,
        help='Path to Haskell definition to use.',
    )
    kcfg_show_subparser.add_argument(
        'spec_file',
        type=file_path,
        help='Path to .k file',
    )
    kcfg_show_subparser.add_argument(
        'spec_module',
        type=str,
        help='Spec main module',
    )
    kcfg_show_subparser.add_argument(
        'claim_id',
        type=str,
        help='Claim id',
    )

    # KCFG to dot
    kcfg_to_dot_subparser = command_parser.add_parser(
        'kcfg-to-dot', help='Dump the given CFG for the proof as DOT for visualization.', parents=[shared_args]
    )
    kcfg_to_dot_subparser.add_argument(
        '--definition-dir',
        dest='definition_dir',
        type=dir_path,
        help='Path to Haskell definition to use.',
    )
    kcfg_to_dot_subparser.add_argument(
        'spec_file',
        type=file_path,
        help='Path to .k file',
    )
    kcfg_to_dot_subparser.add_argument(
        'spec_module',
        type=str,
        help='Spec main module',
    )
    kcfg_to_dot_subparser.add_argument(
        'claim_id',
        type=str,
        help='Claim id',
    )

    # KCFG view
    kcfg_view_subparser = command_parser.add_parser('view-kcfg', help='Display tree view of CFG', parents=[shared_args])
    kcfg_view_subparser.add_argument(
        '--definition-dir',
        dest='definition_dir',
        type=dir_path,
        help='Path to Haskell definition to use.',
    )
    kcfg_view_subparser.add_argument(
        'spec_file',
        type=file_path,
        help='Path to .k file',
    )
    kcfg_view_subparser.add_argument(
        'spec_module',
        type=str,
        help='Spec main module',
    )
    kcfg_view_subparser.add_argument(
        'claim_id',
        type=str,
        help='Claim id',
    )

    # show refutation
    show_refutation_subparser = command_parser.add_parser(
        'show-refutation', help='Display the equality proof of a node refutation', parents=[shared_args]
    )
    show_refutation_subparser.add_argument(
        '--definition-dir',
        dest='definition_dir',
        type=dir_path,
        help='Path to Haskell definition to use.',
    )
    show_refutation_subparser.add_argument(
        '--node',
        dest='node',
        type=str,
        help='node short hash',
    )

    return parser


def _loglevel(args: Namespace) -> int:
    if args.debug:
        return logging.DEBUG

    if args.verbose:
        return logging.INFO

    return logging.WARNING


if __name__ == '__main__':
    main()
