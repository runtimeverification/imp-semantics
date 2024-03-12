from __future__ import annotations

import logging
import subprocess
from argparse import ArgumentParser
from pathlib import Path
import os
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Any, Final

from pyk.cli.utils import dir_path, file_path
from pyk.ktool.kprint import KAstOutput, gen_glr_parser
from pyk.ktool.krun import KRunOutput

from .kimp import KIMP

if TYPE_CHECKING:
    from argparse import Namespace

_LOGGER: Final = logging.getLogger(__name__)
_LOG_FORMAT: Final = '%(levelname)s %(asctime)s %(name)s - %(message)s'


def find_definiton_dir(target: str) -> Path:
    '''
    Find the kompiled definiton directory for a `kbuild` target target:
    * if the KIMP_${target.upper}_DIR is set --- use that
    * otherwise ask `kbuild`
    '''

    def kbuild_definition_dir(target: str) -> Path:
        proc_result = subprocess.run(
            ['poetry', 'run', 'kbuild', 'which', target],
            capture_output=True,
        )
        if proc_result.returncode:
            _LOGGER.critical(
                f'Could not find kbuild definition for target {target}. Run kbuild kompile {target}, or specify --definition.'
            )
            exit(proc_result.returncode)
        else:
            return Path(proc_result.stdout.splitlines()[0].decode())

    env_definition_dir = os.environ.get(f'KIMP_{target.upper()}_DIR')
    if env_definition_dir:
        path = Path(env_definition_dir).resolve()
        _LOGGER.info(f'Using kompiled definiton at {str(path)}')
        return path
    else:
        return kbuild_definition_dir(target)


def find_k_src_dir() -> Path:
    '''
    A heuristic way to find the the k-src dir with the K sources is located:
    * if KIMP_K_SRC environment variable is set --- use that
    * otherwise, use ./k-src and hope it works
    '''
    ksrc_dir = os.environ.get(f'KIMP_K_SRC')
    if ksrc_dir:
        ksrc_dir = Path(ksrc_dir).resolve()
    else:
        ksrc_dir = Path('./k-src')
    return ksrc_dir


def main() -> None:
    parser = create_argument_parser()
    args = parser.parse_args()
    logging.basicConfig(level=_loglevel(args), format=_LOG_FORMAT)

    executor_name = 'exec_' + args.command.lower().replace('-', '_')
    if executor_name not in globals():
        raise AssertionError(f'Unimplemented command: {args.command}')

    execute = globals()[executor_name]
    execute(**vars(args))


def exec_run(
    input_file: str,
    definition_dir: str,
    input_term: str | None = None,
    output: str = 'none',
    ignore_return_code: bool = False,
    **kwargs: Any,
) -> None:
    krun_output = KRunOutput[output.upper()]

    imp_parser = None
    if definition_dir is None:
        definition_dir_path = find_definiton_dir('llvm')
        imp_parser = definition_dir_path / 'parser_Stmt_STATEMENTS-SYNTAX'
        if not imp_parser.is_file():
            imp_parser = gen_glr_parser(
                imp_parser, definition_dir=definition_dir_path, module='STATEMENTS-SYNTAX', sort='Stmt'
            )
    else:
        definition_dir_path = Path(definition_dir)
    kimp = KIMP(definition_dir_path, definition_dir_path, imp_parser)

    try:
        with NamedTemporaryFile(mode='w') as f:
            temp_file = Path(f.name)
            if input_term is not None:
                temp_file.write_text(input_term)
            else:
                temp_file.write_text(Path(input_file).read_text())
            proc_res = kimp.run_program(temp_file, output=krun_output)
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
    max_iterations: int,
    max_depth: int,
    ignore_return_code: bool = False,
    reinit: bool = False,
    **kwargs: Any,
) -> None:
    if definition_dir is None:
        definition_dir = str(find_definiton_dir('haskell'))
    k_src_dir = str(find_k_src_dir())

    kimp = KIMP(definition_dir, definition_dir)

    try:
        kimp.prove(
            spec_file=spec_file,
            spec_module=spec_module,
            claim_id=claim_id,
            max_iterations=max_iterations,
            max_depth=max_depth,
            includes=[k_src_dir],
            reinit=reinit,
        )
    except ValueError as err:
        _LOGGER.critical(err.args)
        # exit(1)
        raise
    except RuntimeError as err:
        if ignore_return_code:
            msg, stdout, stderr = err.args
            print(stdout)
            print(stderr)
            print(msg)
        else:
            raise


def exec_show(
    definition_dir: str,
    spec_module: str,
    claim_id: str,
    **kwargs: Any,
) -> None:
    definition_dir = str(find_definiton_dir('haskell'))
    kimp = KIMP(definition_dir, definition_dir)
    kimp.show_kcfg(spec_module, claim_id)


def exec_view(
    definition_dir: str,
    spec_module: str,
    claim_id: str,
    **kwargs: Any,
) -> None:
    definition_dir = str(find_definiton_dir('haskell'))
    kimp = KIMP(definition_dir, definition_dir)
    kimp.view_kcfg(spec_module, claim_id)


def create_argument_parser() -> ArgumentParser:
    # args shared by all commands
    shared_args = ArgumentParser(add_help=False)
    shared_args.add_argument('--verbose', '-v', default=False, action='store_true', help='Verbose output.')
    shared_args.add_argument('--debug', default=False, action='store_true', help='Debug output.')
    shared_args.add_argument(
        '--definition',
        dest='definition_dir',
        nargs='?',
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
        default=1000,
        type=int,
        help='Store every Nth state in the CFG for inspection.',
    )

    parser = ArgumentParser(prog='kimp', description='KIMP command line tool')
    command_parser = parser.add_subparsers(dest='command', required=True, help='Command to execute')

    # Run
    run_subparser = command_parser.add_parser('run', help='Run an IMP program', parents=[shared_args])
    input_group = run_subparser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--input-file',
        type=file_path,
        help='Path to .imp file',
    )
    input_group.add_argument(
        '--input-term',
        dest='input_term',
        type=str,
        help='Program to run, as a literal string',
    )
    run_subparser.add_argument(
        '--output',
        dest='output',
        type=str,
        default='pretty',
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
    _ = command_parser.add_parser(
        'prove', help='Prove a K claim', parents=[shared_args, spec_file_shared_args, claim_shared_args, explore_args]
    )

    # KCFG show
    command_parser.add_parser(
        'show', help="Display a proof's symbolic execution tree as text", parents=[shared_args, claim_shared_args]
    )
    # KCFG view
    command_parser.add_parser(
        'view',
        help="Display a proof's symbolic execution tree in an intercative viewver",
        parents=[shared_args, claim_shared_args],
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
