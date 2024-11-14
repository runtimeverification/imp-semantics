from __future__ import annotations

import logging
from argparse import ArgumentParser
from typing import TYPE_CHECKING, Any, Final

from pyk.cli.utils import file_path

from .kimp import KImp

if TYPE_CHECKING:
    from argparse import Namespace
    from pathlib import Path

_LOGGER: Final = logging.getLogger(__name__)
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


def exec_run(
    input_file: Path,
    env_list: list[list[tuple[str, int]]] | None,
    depth: int | None = None,
    **kwargs: Any,
) -> None:
    kimp = KImp()
    pgm = input_file.read_text()
    env = {var: val for assign in env_list for var, val in assign} if env_list else {}
    pattern = kimp.pattern(pgm=pgm, env=env)
    output = kimp.run(pattern, depth=depth)
    print(kimp.pretty(output, color=True))


def exec_prove(
    spec_file: str,
    spec_module: str,
    claim_id: str,
    max_iterations: int,
    max_depth: int,
    reinit: bool = False,
    **kwargs: Any,
) -> None:
    kimp = KImp()
    kimp.prove(
        spec_file=spec_file,
        spec_module=spec_module,
        claim_id=claim_id,
        max_iterations=max_iterations,
        max_depth=max_depth,
        reinit=reinit,
    )


def exec_show(
    spec_module: str,
    claim_id: str,
    **kwargs: Any,
) -> None:
    kimp = KImp()
    kimp.show_kcfg(spec_module, claim_id)


def exec_view(
    spec_module: str,
    claim_id: str,
    **kwargs: Any,
) -> None:
    kimp = KImp()
    kimp.view_kcfg(spec_module, claim_id)


def create_argument_parser() -> ArgumentParser:
    # args shared by all commands
    shared_args = ArgumentParser(add_help=False)
    shared_args.add_argument('--verbose', '-v', default=False, action='store_true', help='Verbose output.')
    shared_args.add_argument('--debug', default=False, action='store_true', help='Debug output.')

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
        default=False,
        action='store_true',
        help='Reinitialize proof even if it already exists.',
    )
    explore_args.add_argument(
        '--max-depth',
        default=100,
        type=int,
        help='Max depth of execution',
    )
    explore_args.add_argument(
        '--max-iterations',
        default=1000,
        type=int,
        help='Store every Nth state in the CFG for inspection.',
    )

    parser = ArgumentParser(prog='kimp', description='KImp command line tool')
    command_parser = parser.add_subparsers(dest='command', required=True, help='Command to execute')

    # Run
    def env(s: str) -> list[tuple[str, int]]:
        return [(var.strip(), int(val)) for var, val in (assign.split('=') for assign in s.split(','))]

    run_subparser = command_parser.add_parser('run', help='Run an IMP program', parents=[shared_args])
    run_subparser.add_argument('input_file', metavar='INPUT_FILE', type=file_path, help='Path to .imp file')
    run_subparser.add_argument(
        '--env',
        dest='env_list',
        action='append',
        type=env,
        help='Assigments of initial values in form x=0,y=1,...',
    )
    run_subparser.add_argument(
        '--depth',
        type=int,
        help='Execute at most DEPTH rewrite steps',
    )

    # Prove
    command_parser.add_parser(
        'prove', help='Prove a K claim', parents=[shared_args, spec_file_shared_args, claim_shared_args, explore_args]
    )

    # KCFG show
    command_parser.add_parser(
        'show', help="Display a proof's symbolic execution tree as text", parents=[shared_args, claim_shared_args]
    )

    # KCFG view
    command_parser.add_parser(
        'view',
        help="Display a proof's symbolic execution tree in an intercative viewer",
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
