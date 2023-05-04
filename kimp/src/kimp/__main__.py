from __future__ import annotations

import logging
import subprocess
from argparse import ArgumentParser
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from pyk.cli_utils import dir_path, file_path
from pyk.ktool.kprint import KAstInput, KAstOutput
from pyk.ktool.krun import KRunOutput

from .kimp import KIMP

if TYPE_CHECKING:
    from argparse import Namespace

_LOGGER: Final = logging.getLogger(__name__)
_LOG_FORMAT: Final = '%(levelname)s %(asctime)s %(name)s - %(message)s'


def kbuild_definition_dir(target: str) -> Path:
    proc_result = subprocess.run(
        ['poetry', 'run', 'kbuild', 'which', target],
        capture_output=True,
    )
    if proc_result.returncode:
        _LOGGER.critical(
            f'Could not find kbuild definition for target {target}. Run kbuild kompile {target}, or specify --definition-dir.'
        )
        exit(proc_result.returncode)
    else:
        return Path(proc_result.stdout.splitlines()[0].decode())


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
    max_iterations: int,
    max_depth: int,
    reinit: bool,
    ignore_return_code: bool = False,
    # output: str = 'none',
    **kwargs: Any,
) -> None:
    kimp = KIMP(definition_dir, definition_dir)

    try:
        kimp.prove(
            spec_file=spec_file,
            spec_module=spec_module,
            claim_id=claim_id,
            max_iterations=max_iterations,
            max_depth=max_depth,
            reinit=reinit,
        )
    except ValueError as err:
        _LOGGER.critical(err.args)
        exit(1)
    except RuntimeError as err:
        if ignore_return_code:
            msg, stdout, stderr = err.args
            print(stdout)
            print(stderr)
            print(msg)
        else:
            raise


def exec_summarize(
    definition_dir: str,
    spec_file: str,
    spec_module: str,
    claim_id: str,
    max_iterations: int = 20,
    ignore_return_code: bool = False,
    # output: str = 'none',
    **kwargs: Any,
) -> None:
    kimp = KIMP(definition_dir, definition_dir)

    try:
        kimp.summarize(spec_file=spec_file, spec_module=spec_module, claim_id=claim_id, max_iterations=max_iterations)
    except ValueError as err:
        _LOGGER.critical(err.args)
        exit(1)
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
    max_iterations: int,
    max_depth: int,
    reinit: bool,
    bmc_depth: int,
    # output: str = 'none',
    ignore_return_code: bool = False,
    **kwargs: Any,
) -> None:
    kimp = KIMP(definition_dir, definition_dir)

    try:
        kimp.bmc_prove(
            spec_file=spec_file,
            spec_module=spec_module,
            claim_id=claim_id,
            max_iterations=max_iterations,
            max_depth=max_depth,
            reinit=reinit,
            bmc_depth=bmc_depth,
        )
    except ValueError as err:
        _LOGGER.critical(err.args)
        exit(1)
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
    spec_module: str,
    claim_id: str,
    node: str,
    # output: str = 'none',
    ignore_return_code: bool = False,
    **kwargs: Any,
) -> None:
    kimp = KIMP(definition_dir, definition_dir)

    try:
        kimp.kcfg_refute_node(spec_module=spec_module, claim_id=claim_id, node_short_hash=node)
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
    spec_module: str,
    claim_id: str,
    to_module: bool = False,
    inline_nodes: bool = False,
    **kwargs: Any,
) -> None:
    kimp = KIMP(definition_dir, definition_dir)
    kimp.show_kcfg(spec_module, claim_id, to_module=to_module, inline_nodes=inline_nodes)


def exec_view_kcfg(
    definition_dir: str,
    spec_module: str,
    claim_id: str,
    **kwargs: Any,
) -> None:
    kimp = KIMP(definition_dir, definition_dir)
    kimp.view_kcfg(spec_module, claim_id)


def exec_show_refutation(
    definition_dir: str,
    spec_module: str,
    claim_id: str,
    node: str,
    **kwargs: Any,
) -> None:
    kimp = KIMP(definition_dir, definition_dir)
    kimp.show_refutation(spec_module, claim_id, node=node)


def exec_kcfg_to_dot(
    definition_dir: str,
    spec_module: str,
    claim_id: str,
    **kwargs: Any,
) -> None:
    kimp = KIMP(definition_dir, definition_dir)
    kimp.kcfg_to_dot(spec_module, claim_id)


def create_argument_parser() -> ArgumentParser:
    # args shared by all commands
    shared_args = ArgumentParser(add_help=False)
    shared_args.add_argument('--verbose', '-v', default=False, action='store_true', help='Verbose output.')
    shared_args.add_argument('--debug', default=False, action='store_true', help='Debug output.')
    shared_args.add_argument(
        '--definition-dir',
        dest='definition_dir',
        nargs='?',
        default=kbuild_definition_dir('haskell'),
        type=dir_path,
        help='Path to compiled K definition to use.',
    )

    # args shared by proof/prover/kcfg commands
    spec_file_shared_args = ArgumentParser(add_help=False)
    spec_file_shared_args.add_argument(
        'spec_file',
        type=file_path,
        help='Path to K spec file',
    )

    claim_shared_args = ArgumentParser(add_help=False)
    claim_shared_args.add_argument(
        'spec_module',
        type=str,
        help='Spec main module',
    )
    claim_shared_args.add_argument(
        'claim_id',
        type=str,
        help='Claim id',
    )

    explore_args = ArgumentParser(add_help=False)
    explore_args.add_argument(
        '--reinit',
        dest='reinit',
        default=False,
        action='store_true',
        help='Reinitialize proof even if it already exists.',
    )
    explore_args.add_argument(
        '--max-depth',
        dest='max_depth',
        default=100,
        type=int,
        help='Max depth of execution',
    )
    explore_args.add_argument(
        '--max-iterations',
        dest='max_iterations',
        default=20,
        type=int,
        help='Store every Nth state in the CFG for inspection.',
    )

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
    command_parser.add_parser(
        'prove', help='Prove a K claim', parents=[shared_args, spec_file_shared_args, claim_shared_args, explore_args]
    )
    prove_subparser.add_argument(
        '--max-iterations',
        type=int,
        default=20,
        help='Maximum number of iterations to run prover for.',
    )

    # Summarize
    summarize_subparser = command_parser.add_parser('summarize', help='Prove a K claim', parents=[shared_args])
    summarize_subparser.add_argument(
        '--definition-dir',
        dest='definition_dir',
        type=dir_path,
        help='Path to Haskell definition to use.',
    )
    summarize_subparser.add_argument(
        'spec_file',
        type=file_path,
        help='Path to .k file',
    )
    summarize_subparser.add_argument(
        'spec_module',
        type=str,
        help='Spec main module',
    )
    summarize_subparser.add_argument(
        'claim_id',
        type=str,
        help='Claim id',
    )
    summarize_subparser.add_argument(
        '--max-iterations',
        type=int,
        default=20,
        help='Maximum number of iterations to run summarizer for.',
    )

    # BMC Prove
    bmc_prove_subparser = command_parser.add_parser(
        'bmc-prove',
        help='Prove a K claim with the Bounded Model-Checker',
        parents=[shared_args, spec_file_shared_args, claim_shared_args, explore_args],
    )
    bmc_prove_subparser.add_argument(
        '--bmc-depth',
        type=int,
        default=1,
        help='Model checking bound',
    )

    # Refute node
    refute_node_subparser = command_parser.add_parser(
        'refute-node', help='Refute a node as infeasible', parents=[shared_args, claim_shared_args]
    )
    refute_node_subparser.add_argument(
        '--node',
        dest='node',
        type=str,
        help='node short hash',
    )

    # show refutation
    show_refutation_subparser = command_parser.add_parser(
        'show-refutation',
        help='Display the equality proof of a node refutation',
        parents=[shared_args, claim_shared_args],
    )
    show_refutation_subparser.add_argument(
        '--node',
        dest='node',
        type=str,
        help='node short hash',
    )

    # EQ prove
    eq_prove_subparser = command_parser.add_parser('eq-prove', help='Prove an equality', parents=[shared_args])
    eq_prove_subparser.add_argument(
        'proof_id',
        type=str,
        help='Id of a JSON-serialized proof',
    )

    # KCFG show
    kcfg_show_subparser = command_parser.add_parser(
        'show-kcfg', help='Display tree show of CFG', parents=[shared_args, claim_shared_args]
    )
    kcfg_show_subparser.add_argument(
        '--to-module',
        default=False,
        action='store_true',
        help='Display a K module containing the KCFG thus far.',
    )
    kcfg_show_subparser.add_argument(
        '--inline-nodes',
        default=False,
        action='store_true',
        help='Display states inline with KCFG nodes.',
    )
    # KCFG to dot
    command_parser.add_parser(
        'kcfg-to-dot',
        help='Dump the given CFG for the proof as DOT for visualization.',
        parents=[shared_args, claim_shared_args],
    )

    # KCFG view
    command_parser.add_parser('view-kcfg', help='Display tree view of CFG', parents=[shared_args, claim_shared_args])

    return parser


def _loglevel(args: Namespace) -> int:
    if args.debug:
        return logging.DEBUG

    if args.verbose:
        return logging.INFO

    return logging.WARNING


if __name__ == '__main__':
    main()
