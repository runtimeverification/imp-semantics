from __future__ import annotations

import logging
import os
from argparse import ArgumentParser
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from pyk.cli.utils import dir_path, file_path

from .kimp import KImp

if TYPE_CHECKING:
    from argparse import Namespace

_LOGGER: Final = logging.getLogger(__name__)
_LOG_FORMAT: Final = '%(levelname)s %(asctime)s %(name)s - %(message)s'


def find_target(target: str) -> Path:
    """
    Find a `kdist` target:
    * if KIMP_${target.upper}_DIR is set --- use that
    * otherwise ask `kdist`
    """

    env_target_dir = os.environ.get(f'KIMP_{target.upper()}_DIR')
    if env_target_dir:
        path = Path(env_target_dir).resolve()
        _LOGGER.info(f'Using target at {path}')
        return path
    else:
        from pyk.kdist import kdist

        return kdist.which(f'imp-semantics.{target}')


def find_k_src_dir() -> Path:
    """
    A heuristic way to find the the k-src dir with the K sources is located:
    * if KIMP_K_SRC environment variable is set --- use that
    * otherwise, use ./k-src and hope it works
    """
    ksrc_dir_str = os.environ.get('KIMP_K_SRC')
    if ksrc_dir_str is not None:
        ksrc_dir = Path(ksrc_dir_str).resolve()
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
    input_file: Path,
    definition_dir: Path | None,
    depth: int | None = None,
    **kwargs: Any,
) -> None:
    definition_dir = find_target('llvm') if definition_dir is None else definition_dir
    kimp = KImp(definition_dir, definition_dir)
    pgm = input_file.read_text()
    pattern = kimp.pattern(pgm=pgm, env={})
    output = kimp.run(pattern, depth=depth)
    print(kimp.pretty(output))


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
        definition_dir = str(find_target('haskell'))
    k_src_dir = str(find_target('source') / 'imp-semantics')

    kimp = KImp(definition_dir, definition_dir)

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
    definition_dir = str(find_target('haskell'))
    kimp = KImp(definition_dir, definition_dir)
    kimp.show_kcfg(spec_module, claim_id)


def exec_view(
    definition_dir: str,
    spec_module: str,
    claim_id: str,
    **kwargs: Any,
) -> None:
    definition_dir = str(find_target('haskell'))
    kimp = KImp(definition_dir, definition_dir)
    kimp.view_kcfg(spec_module, claim_id)


def create_argument_parser() -> ArgumentParser:
    # args shared by all commands
    shared_args = ArgumentParser(add_help=False)
    shared_args.add_argument('--verbose', '-v', default=False, action='store_true', help='Verbose output.')
    shared_args.add_argument('--debug', default=False, action='store_true', help='Debug output.')
    shared_args.add_argument(
        '--definition',
        dest='definition_dir',
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

    parser = ArgumentParser(prog='kimp', description='KImp command line tool')
    command_parser = parser.add_subparsers(dest='command', required=True, help='Command to execute')

    # Run
    run_subparser = command_parser.add_parser('run', help='Run an IMP program', parents=[shared_args])
    run_subparser.add_argument('input_file', metavar='INPUT_FILE', type=file_path, help='Path to .imp file')
    run_subparser.add_argument(
        '--depth',
        dest='depth',
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
