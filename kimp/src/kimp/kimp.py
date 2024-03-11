from __future__ import annotations
from pyk.kast.manip import remove_generated_cells, get_cell
from pyk.kast.pretty import SymbolTable, paren
from pyk.kcfg.kcfg import KCFG

from pyk.kcfg.show import NodePrinter

__all__ = ['KIMP']

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Iterable, Iterator, Optional, Union, final

from pyk.cli.utils import check_dir_path, check_file_path
from pyk.cterm.cterm import CTerm
from pyk.cterm.symbolic import CTermSymbolic
from pyk.kast.inner import KApply, KSequence, KVariable
from pyk.kast.manip import ml_pred_to_bool
from pyk.kcfg.explore import KCFGExplore
from pyk.kcfg.semantics import KCFGSemantics
from pyk.kore.kompiled import KompiledKore
from pyk.kore.rpc import KoreClient, kore_server
from pyk.ktool.kprint import gen_glr_parser
from pyk.ktool.kprove import KProve
from pyk.ktool.krun import KRun, KRunOutput, _krun
from pyk.prelude.kbool import BOOL, notBool
from pyk.prelude.ml import mlAnd, mlEqualsTrue
from pyk.proof.reachability import APRProof, APRProver
from pyk.proof.show import APRProofNodePrinter, APRProofShow
from pyk.proof.tui import APRProofViewer
from pyk.utils import single

if TYPE_CHECKING:
    from subprocess import CompletedProcess
    from typing import Final

    from pyk.kast.inner import KInner
    from pyk.kast.outer import KDefinition
    from pyk.kore.rpc import FallbackReason
    from pyk.ktool.kprint import KPrint
    from pyk.utils import BugReport


_LOGGER: Final = logging.getLogger(__name__)


class ImpSemantics(KCFGSemantics):
    definition: KDefinition | None

    def __init__(self, definition: KDefinition | None = None):
        super().__init__()
        self.definition = definition

    def is_terminal(self, c: CTerm) -> bool:
        k_cell = c.cell('K_CELL')
        if type(k_cell) is KSequence:
            if len(k_cell) == 0:
                return True
            if len(k_cell) == 1 and type(k_cell[0]) is KVariable:
                return True
        if type(k_cell) is KVariable:
            return True
        return False

    def extract_branches(self, c: CTerm) -> list[KInner]:
        if self.definition is None:
            raise ValueError('IMP branch extraction requires a non-None definition')

        k_cell = c.cell('K_CELL')
        if type(k_cell) is KSequence and len(k_cell) > 0:
            k_cell = k_cell[0]
        if type(k_cell) is KApply and k_cell.label.name == 'if(_)_else_':
            condition = k_cell.args[0]
            if (type(condition) is KVariable and condition.sort == BOOL) or (
                type(condition) is KApply and self.definition.return_sort(condition.label) == BOOL
            ):
                return [mlEqualsTrue(condition), mlEqualsTrue(notBool(condition))]
        return []

    def abstract_node(self, c: CTerm) -> CTerm:
        return c

    def same_loop(self, c1: CTerm, c2: CTerm) -> bool:
        k_cell_1 = c1.cell('K_CELL')
        k_cell_2 = c2.cell('K_CELL')
        if k_cell_1 == k_cell_2 and type(k_cell_1) is KSequence and type(k_cell_1[0]) is KApply:
            return k_cell_1[0].label.name == 'while(_)_'  # type: ignore
        return False


@final
@dataclass(frozen=True)
class KIMP:
    llvm_dir: Path
    haskell_dir: Path
    imp_parser: Path
    proof_dir: Path

    def __init__(self, llvm_dir: Union[str, Path], haskell_dir: Union[str, Path]):
        llvm_dir = Path(llvm_dir)
        check_dir_path(llvm_dir)

        imp_parser = llvm_dir / 'parser_Stmt_STATEMENTS-SYNTAX'
        if not imp_parser.is_file():
            imp_parser = gen_glr_parser(imp_parser, definition_dir=llvm_dir, module='STATEMENTS-SYNTAX', sort='Stmt')

        haskell_dir = Path(haskell_dir)
        check_dir_path(haskell_dir)

        proof_dir = Path('.') / '.kimp' / 'proofs'
        proof_dir.mkdir(exist_ok=True, parents=True)

        object.__setattr__(self, 'llvm_dir', llvm_dir)
        object.__setattr__(self, 'haskell_dir', haskell_dir)
        object.__setattr__(self, 'imp_parser', imp_parser)
        object.__setattr__(self, 'proof_dir', proof_dir)

    @cached_property
    def kprove(self) -> KProve:
        kprove = KProve(
            definition_dir=self.haskell_dir, use_directory=self.proof_dir, patch_symbol_table=KIMP._patch_symbol_table
        )
        return kprove

    @cached_property
    def krun(self) -> KRun:
        krun = KRun(definition_dir=self.llvm_dir)
        return krun

    def run_program(
        self,
        program_file: Union[str, Path],
        *,
        output: KRunOutput = KRunOutput.NONE,
        check: bool = True,
        temp_file: Optional[Union[str, Path]] = None,
    ) -> CompletedProcess:
        def run(program_file: Path) -> CompletedProcess:
            return _krun(
                input_file=program_file,
                definition_dir=self.llvm_dir,
                output=output,
                check=check,
                pipe_stderr=True,
                pmap={'PGM': str(self.imp_parser)},
            )

        def preprocess_and_run(program_file: Path, temp_file: Path) -> CompletedProcess:
            temp_file.write_text(program_file.read_text())
            return run(temp_file)

        program_file = Path(program_file)
        check_file_path(program_file)

        if temp_file is None:
            with NamedTemporaryFile(mode='w') as f:
                temp_file = Path(f.name)
                return preprocess_and_run(program_file, temp_file)

        temp_file = Path(temp_file)
        return preprocess_and_run(program_file, temp_file)

    def prove(
        self,
        spec_file: str,
        spec_module: str,
        includes: Iterable[str],
        claim_id: str,
        max_iterations: int,
        max_depth: int,
        reinit: bool,
    ) -> None:
        include_dirs = [Path(include) for include in includes]

        claims = self.kprove.get_claims(
            Path(spec_file), spec_module_name=spec_module, claim_labels=[claim_id], include_dirs=include_dirs
        )
        claim = single(claims)
        spec_label = f'{spec_module}.{claim_id}'

        if not reinit and APRProof.proof_data_exists(spec_label, self.proof_dir):
            # load an existing proof (to continue work in it)
            proof = APRProof.read_proof_data(proof_dir=self.proof_dir, id=f'{spec_module}.{claim_id}')
        else:
            # ignore existing proof data and reinitilize it from a claim
            proof = APRProof.from_claim(self.kprove.definition, claim=claim, logs={}, proof_dir=self.proof_dir)

        with legacy_explore(
            self.kprove,
            kcfg_semantics=ImpSemantics(self.kprove.definition),
            id=spec_label,
        ) as kcfg_explore:
            prover = APRProver(proof, kcfg_explore=kcfg_explore, execute_depth=max_depth, cut_point_rules=['IMP.while'])
            prover.advance_proof(max_iterations=max_iterations)

            print(proof.summary)
            print('============================================')
            print("What's next?: ")
            print('============================================')
            print('To inspect the symbolic execution trace interactively, run: ')
            print(f'  kimp view-kcfg {spec_module} {claim_id}')
            print('============================================')
            print('To dump the symbolic execution trace into stdout, run: ')
            print(f'  kimp show-kcfg {spec_module} {claim_id}')
            print('============================================')
            if not proof.passed:
                print('To retry the failed/pending proof, run : ')
                print(f'  kimp prove {spec_file} {spec_module} {claim_id}')
            print('To start the proof from scratch: ')
            print(f'  kimp prove --reinit {spec_file} {spec_module} {claim_id}')

    def view_kcfg(
        self,
        spec_module: str,
        claim_id: str,
    ) -> None:
        proof = APRProof.read_proof_data(proof_dir=self.proof_dir, id=f'{spec_module}.{claim_id}')
        kcfg_viewer = APRProofViewer(proof, self.kprove, node_printer=KIMPNodePrinter(kimp=self))
        kcfg_viewer.run()

    def show_kcfg(
        self,
        spec_module: str,
        claim_id: str,
    ) -> None:
        proof = APRProof.read_proof_data(proof_dir=self.proof_dir, id=f'{spec_module}.{claim_id}')
        proof_show = APRProofShow(self.kprove, node_printer=KIMPNodePrinter(kimp=self))
        res_lines = proof_show.show(
            proof,
        )
        print('\n'.join(res_lines))

    @classmethod
    def _patch_symbol_table(cls, symbol_table: SymbolTable) -> None:
        symbol_table['_Map_'] = paren(lambda m1, m2: m1 + '\n' + m2)


@contextmanager
def legacy_explore(
    kprint: KPrint,
    *,
    kcfg_semantics: KCFGSemantics | None = None,
    id: str | None = None,
    port: int | None = None,
    kore_rpc_command: str | Iterable[str] | None = None,
    llvm_definition_dir: Path | None = None,
    smt_timeout: int | None = None,
    smt_retry_limit: int | None = None,
    smt_tactic: str | None = None,
    bug_report: BugReport | None = None,
    log_axioms_file: Path | None = None,
    start_server: bool = True,
    fallback_on: Iterable[FallbackReason] | None = None,
    interim_simplification: int | None = None,
    no_post_exec_simplify: bool = False,
) -> Iterator[KCFGExplore]:
    if start_server:
        with kore_server(
            definition_dir=kprint.definition_dir,
            llvm_definition_dir=llvm_definition_dir,
            module_name=kprint.main_module,
            port=port,
            command=kore_rpc_command,
            bug_report=bug_report,
            smt_timeout=smt_timeout,
            smt_retry_limit=smt_retry_limit,
            smt_tactic=smt_tactic,
            log_axioms_file=log_axioms_file,
            fallback_on=fallback_on,
            interim_simplification=interim_simplification,
            no_post_exec_simplify=no_post_exec_simplify,
        ) as server:
            with KoreClient('localhost', server.port, bug_report=bug_report, bug_report_id=id) as client:
                cterm_symbolic = cterm_symbolic = CTermSymbolic(
                    kore_client=client,
                    definition=kprint.definition,
                    kompiled_kore=KompiledKore.load(kprint.definition_dir),
                )
                yield KCFGExplore(
                    # kore_client=client,
                    kcfg_semantics=kcfg_semantics,
                    id=id,
                    cterm_symbolic=cterm_symbolic,
                )
    else:
        if port is None:
            raise ValueError('Missing port with start_server=False')
        with KoreClient('localhost', port, bug_report=bug_report, bug_report_id=id) as client:
            cterm_symbolic = cterm_symbolic = CTermSymbolic(
                kore_client=client,
                definition=kprint.definition,
                kompiled_kore=KompiledKore.load(kprint.definition_dir),
            )
            yield KCFGExplore(
                kcfg_semantics=kcfg_semantics,
                id=id,
                cterm_symbolic=cterm_symbolic,
            )


class KIMPNodePrinter(NodePrinter):
    kimp: KIMP

    def __init__(self, kimp: KIMP):
        NodePrinter.__init__(self, kimp.kprove)
        self.kimp = kimp

    def print_node(self, kcfg: KCFG, node: KCFG.Node) -> list[str]:
        ret_strs = super().print_node(kcfg, node)
        config = get_cell(remove_generated_cells(node.cterm.config), "K_CELL")
        env = get_cell(remove_generated_cells(node.cterm.config), "ENV_CELL")
        # pretty-print the configuration
        ret_strs += self.kimp.kprove.pretty_print(config).splitlines()
        ret_strs += ["env:"]
        ret_strs += ["  " + l.replace("( ", "").replace(" )", "") for l in self.kimp.kprove.pretty_print(env).splitlines()]
        # pretty-print the constraints
        constraints = [ml_pred_to_bool(c) for c in node.cterm.constraints]
        if len(constraints) > 0:
            ret_strs += ["constraints:"]
            for c in constraints:
                ret_strs.append("  " + self.kimp.kprove.pretty_print(c))
        return ret_strs
