from __future__ import annotations
from pyk.kast.outer import KDefinition

from pyk.kcfg.semantics import KCFGSemantics

__all__ = ['KIMP']

import logging
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Any, Iterable, Optional, Union, final

from pyk.cli_utils import check_dir_path, check_file_path
from pyk.kast.inner import KApply, KSequence, KVariable
from pyk.kcfg.explore import KCFGExplore
from pyk.kcfg.kcfg import KCFG
from pyk.kcfg.tui import KCFGViewer
from pyk.ktool.kprove import KProve
from pyk.ktool.krun import KRun, KRunOutput, _krun
from pyk.ktool.kprint import gen_glr_parser
from pyk.prelude.kbool import BOOL, notBool
from pyk.prelude.ml import mlEqualsTrue
from pyk.proof.proof import Proof
from pyk.kast.pretty import paren
from pyk.proof.reachability import APRProof, APRProver

if TYPE_CHECKING:
    from subprocess import CompletedProcess
    from typing import Final
    from pyk.cterm.cterm import CTerm
    from pyk.kast.pretty import SymbolTable
    from pyk.kast.inner import KInner


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

        imp_parser = llvm_dir / 'parser_Stmt_CALLS-SYNTAX'
        if not imp_parser.is_file():
            imp_parser = gen_glr_parser(imp_parser, definition_dir=llvm_dir, module='CALLS-SYNTAX', sort='Stmt')

        haskell_dir = Path(haskell_dir)
        check_dir_path(haskell_dir)

        proof_dir = Path('.') / '.kimp' / 'ag_proofs'
        proof_dir.mkdir(exist_ok=True, parents=True)

        object.__setattr__(self, 'llvm_dir', llvm_dir)
        object.__setattr__(self, 'haskell_dir', haskell_dir)
        object.__setattr__(self, 'imp_parser', imp_parser)
        object.__setattr__(self, 'proof_dir', proof_dir)

    #     @classmethod
    #     def _patch_symbol_table(cls, symbol_table: SymbolTable) -> None:
    #         # fmt: off
    #         symbol_table['while(_)_'] = lambda cond, body: f'\n while({cond})' + '\n' + body
    #         symbol_table['if(_)_else_'] = lambda cond, t, e: f'\n if ({cond})' + '\n' + t + '\n' f'else {e}'
    #         symbol_table['_Map_'] = paren(lambda m1, m2: m1 + '\n' + m2)
    #         paren_symbols = [
    #             '_|->_',
    #             '#And',
    #             '_andBool_',
    #             '#Implies',
    #             '_impliesBool_',
    #             '_&Int_',
    #             '_*Int_',
    #             '_+Int_',
    #             '_-Int_',
    #             '_/Int_',
    #             '_|Int_',
    #             '_modInt_',
    #             'notBool_',
    #             '#Or',
    #             '_orBool_',
    #             '_Set_',
    #         ]
    #         for symb in paren_symbols:
    #             if symb in symbol_table:
    #                 symbol_table[symb] = paren(symbol_table[symb])
    #         # fmt: on

    #     @cached_property
    #     def kprove(self) -> KProve:
    #         kprove = KProve(definition_dir=self.haskell_dir, use_directory=self.proof_dir)
    #         # KIMP._patch_symbol_table(kprove._symbol_table)
    #         return kprove

    @cached_property
    def krun(self) -> KRun:
        krun = KRun(definition_dir=self.llvm_dir)
        # KIMP._patch_symbol_table(krun.symbol_table)
        return krun

    #     @staticmethod
    #     def _is_terminal(cterm1: CTerm) -> bool:
    #         k_cell = cterm1.cell('K_CELL')
    #         if type(k_cell) is KSequence:
    #             if len(k_cell) == 0:
    #                 return True
    #             if len(k_cell) == 1 and type(k_cell[0]) is KVariable:
    #                 return True
    #         if type(k_cell) is KVariable:
    #             return True
    #         return False

    #     def _extract_branches(self, cterm: CTerm) -> Iterable[KInner]:
    #         k_cell = cterm.cell('K_CELL')
    #         if type(k_cell) is KSequence and len(k_cell) > 0:
    #             k_cell = k_cell[0]
    #         if type(k_cell) is KApply and k_cell.label.name == 'if(_)_else_':
    #             condition = k_cell.args[0]
    #             if (type(condition) is KVariable and condition.sort == BOOL) or (
    #                 type(condition) is KApply and self.kprove.definition.return_sort(condition.label) == BOOL
    #             ):
    #                 return [mlEqualsTrue(condition), mlEqualsTrue(notBool(condition))]
    #         return []

    #     @staticmethod
    #     def _same_loop(cterm1: CTerm, cterm2: CTerm) -> bool:
    #         k_cell_1 = cterm1.cell('K_CELL')
    #         k_cell_2 = cterm2.cell('K_CELL')
    #         if k_cell_1 == k_cell_2 and type(k_cell_1) is KSequence and type(k_cell_1[0]) is KApply:
    #             return k_cell_1[0].label.name == 'while(_)_'  # type: ignore
    #         return False

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


#     def prove(
#         self, spec_file: str, spec_module: str, claim_id: str, max_iterations: int, max_depth: int, reinit: bool
#     ) -> None:
#         proof_id = f'{spec_module}.{claim_id}'
#         if Proof.proof_exists(proof_id, proof_dir=self.proof_dir) and not reinit:
#             proof = read_proof(proof_id, proof_dir=self.proof_dir)
#             if type(proof) is not APRProof:
#                 raise ValueError(f'Proof {proof_id} exists and is of type {type(proof)}, while APRProof was expected')
#         else:
#             claims = self.kprove.get_claims(
#                 Path(spec_file),
#                 spec_module_name=spec_module,
#                 claim_labels=[f'{spec_module}.{claim_id}'],
#                 include_dirs=[self.haskell_dir.parent.parent.parent / 'include' / 'imp-semantics'],
#             )
#             assert len(claims) == 1
#             # kcfg = KCFG.from_claim(self.kprove.definition, claims[0])
#             proof = APRProof.from_claim(defn=self.kprove.definition, claim=claims[0], proof_dir=self.proof_dir, logs={})
#         with KCFGExplore(
#             self.kprove,
#             id=f'{spec_module}.{claim_id}',
#         ) as kcfg_explore:
#             prover = APRProver(
#                 proof,
#                 kcfg_explore=kcfg_explore,
#                 cut_point_rules=['IMP.while'],
#             )
#             prover.advance_proof(
#                 kcfg_explore,
#             )

#         proof.write_proof()
#         print('\n'.join(proof.summary.lines))

#     def view_kcfg(
#         self,
#         spec_module: str,
#         claim_id: str,
#         **kwargs: Any,
#     ) -> None:
#         proof = read_proof(f'{spec_module}.{claim_id}', proof_dir=self.proof_dir)
#         if type(proof) is APRProof:
#             kcfg_viewer = KCFGViewer(proof.kcfg, self.kprove)
#             kcfg_viewer.run()
