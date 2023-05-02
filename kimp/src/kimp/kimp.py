from __future__ import annotations

__all__ = ['KIMP']

import json
import subprocess
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from subprocess import CalledProcessError
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Any, Iterable, Optional, Union, final

from pyk.cli_utils import check_dir_path, check_file_path
from pyk.kast.inner import KApply, KInner, KSequence, KVariable
from pyk.kcfg.explore import KCFGExplore
from pyk.kcfg.kcfg import KCFG
from pyk.kcfg.show import KCFGShow
from pyk.kcfg.tui import KCFGViewer
from pyk.ktool.kprint import KAstInput, KAstOutput, _kast, gen_glr_parser, paren
from pyk.ktool.kprove import KProve
from pyk.ktool.krun import KRun, KRunOutput, _krun
from pyk.prelude.kbool import BOOL, notBool
from pyk.prelude.ml import mlEqualsTrue
from pyk.proof.equality import EqualityProof, EqualityProver
from pyk.proof.reachability import APRBMCProof, APRBMCProver, APRProof, APRProver
from pyk.proof.utils import read_proof

if TYPE_CHECKING:
    from subprocess import CompletedProcess

    from pyk.cterm import CTerm
    from pyk.ktool.kprint import SymbolTable


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

        imp_parser = llvm_dir / 'parser_Pgm_IMP-SYNTAX'
        if not imp_parser.is_file():
            imp_parser = gen_glr_parser(imp_parser, definition_dir=llvm_dir, module='IMP-SYNTAX', sort='Pgm')

        haskell_dir = Path(haskell_dir)
        check_dir_path(haskell_dir)

        proof_dir = Path('.') / '.kimp' / 'ag_proofs'
        proof_dir.mkdir(exist_ok=True, parents=True)

        object.__setattr__(self, 'llvm_dir', llvm_dir)
        object.__setattr__(self, 'haskell_dir', haskell_dir)
        object.__setattr__(self, 'imp_parser', imp_parser)
        object.__setattr__(self, 'proof_dir', proof_dir)

    @classmethod
    def _patch_symbol_table(cls, symbol_table: SymbolTable) -> None:
        # fmt: off
        symbol_table['while(_)_'] = lambda cond, body: f'\n while({cond})' + '\n' + body
        symbol_table['if(_)_else_'] = lambda cond, t, e: f'\n if ({cond})' + '\n' + t + '\n' f'else {e}'
        symbol_table['_Map_'] = paren(lambda m1, m2: m1 + '\n' + m2)
        paren_symbols = [
            '_|->_',
            '#And',
            '_andBool_',
            '#Implies',
            '_impliesBool_',
            '_&Int_',
            '_*Int_',
            '_+Int_',
            '_-Int_',
            '_/Int_',
            '_|Int_',
            '_modInt_',
            'notBool_',
            '#Or',
            '_orBool_',
            '_Set_',
        ]
        for symb in paren_symbols:
            if symb in symbol_table:
                symbol_table[symb] = paren(symbol_table[symb])
        # fmt: on

    @cached_property
    def kprove(self) -> KProve:
        kprove = KProve(definition_dir=self.haskell_dir, use_directory=self.proof_dir)
        KIMP._patch_symbol_table(kprove.symbol_table)
        return kprove

    @cached_property
    def krun(self) -> KRun:
        krun = KRun(definition_dir=self.haskell_dir, use_directory=self.proof_dir)
        KIMP._patch_symbol_table(krun.symbol_table)
        return krun

    @staticmethod
    def _is_terminal(cterm1: CTerm) -> bool:
        k_cell = cterm1.cell('K_CELL')
        if type(k_cell) is KSequence:
            if len(k_cell) == 0:
                return True
            if len(k_cell) == 1 and type(k_cell[0]) is KVariable:
                return True
        if type(k_cell) is KVariable:
            return True
        return False

    def _extract_branches(self, cterm: CTerm) -> Iterable[KInner]:
        k_cell = cterm.cell('K_CELL')
        if type(k_cell) is KSequence and len(k_cell) > 0:
            k_cell = k_cell[0]
        if type(k_cell) is KApply and k_cell.label.name == 'if(_)_else_':
            condition = k_cell.args[0]
            if (type(condition) is KVariable and condition.sort == BOOL) or (
                type(condition) is KApply and self.kprove.definition.return_sort(condition.label) == BOOL
            ):
                return [mlEqualsTrue(condition), mlEqualsTrue(notBool(condition))]
        return []

    @staticmethod
    def _same_loop(cterm1: CTerm, cterm2: CTerm) -> bool:
        k_cell_1 = cterm1.cell('K_CELL')
        k_cell_2 = cterm2.cell('K_CELL')
        if k_cell_1 == k_cell_2 and type(k_cell_1) is KSequence and type(k_cell_1[0]) is KApply:
            return k_cell_1[0].label.name == 'while(_)_'
        return False

    def parse_program_raw(
        self,
        program_file: Union[str, Path],
        *,
        input: KAstInput,
        output: KAstOutput,
        temp_file: Optional[Union[str, Path]] = None,
    ) -> CompletedProcess:
        def parse(program_file: Path) -> CompletedProcess:
            try:
                if output == KAstOutput.KORE:
                    command = [str(self.imp_parser)] + [str(program_file)]
                    proc_res = subprocess.run(command, stdout=subprocess.PIPE, check=True, text=True)
                else:
                    proc_res = _kast(
                        definition_dir=self.llvm_dir,
                        file=program_file,
                        input=input,
                        output=output,
                        sort='Pgm',
                    )
            except CalledProcessError as err:
                raise ValueError("Couldn't parse program") from err
            return proc_res

        def preprocess_and_parse(program_file: Path, temp_file: Path) -> CompletedProcess:
            temp_file.write_text(program_file.read_text())
            return parse(temp_file)

        program_file = Path(program_file)
        check_file_path(program_file)

        if temp_file is None:
            with NamedTemporaryFile(mode='w') as f:
                temp_file = Path(f.name)
                return preprocess_and_parse(program_file, temp_file)

        temp_file = Path(temp_file)
        return preprocess_and_parse(program_file, temp_file)

    def parse_program(
        self,
        program_file: Union[str, Path],
        *,
        temp_file: Optional[Union[str, Path]] = None,
    ) -> KInner:
        proc_res = self.parse_program_raw(
            program_file=program_file,
            input=KAstInput.PROGRAM,
            output=KAstOutput.JSON,
            temp_file=temp_file,
        )

        return KInner.from_dict(json.loads(proc_res.stdout)['term'])

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
        claim_id: str,
        # max_iterations: int,
        # max_depth: int,
        # terminal_rules: Iterable[str],
        # cut_rules: Iterable[str],
        # proof_status: ProofStatus,
    ) -> None:
        claims = self.kprove.get_claims(
            Path(spec_file),
            spec_module_name=spec_module,
            claim_labels=[f'{spec_module}.{claim_id}'],
            include_dirs=[self.haskell_dir.parent.parent.parent / 'include' / 'imp-semantics'],
        )
        assert len(claims) == 1

        kcfg = KCFG.from_claim(self.kprove.definition, claims[0])
        proof = APRProof(f'{spec_module}.{claim_id}', kcfg, proof_dir=self.proof_dir)
        prover = APRProver(proof, is_terminal=KIMP._is_terminal, extract_branches=self._extract_branches)
        with KCFGExplore(
            self.kprove,
            id=f'{spec_module}.{claim_id}',
        ) as kcfg_explore:
            kcfg = prover.advance_proof(
                kcfg_explore,
                max_iterations=20,
                # execute_depth=1,
                cut_point_rules=['IMP.while'],
                # terminal_rules='IMP.while',
            )

        proof.write_proof()
        print(proof.status)

    def bmc_prove(
        self,
        spec_file: str,
        spec_module: str,
        claim_id: str,
        # max_iterations: int,
        # max_depth: int,
        # terminal_rules: Iterable[str],
        # cut_rules: Iterable[str],
        # proof_status: ProofStatus,
        bmc_depth: int,
    ) -> None:
        claims = self.kprove.get_claims(
            Path(spec_file),
            spec_module_name=spec_module,
            claim_labels=[f'{spec_module}.{claim_id}'],
            include_dirs=[self.haskell_dir.parent.parent.parent / 'include' / 'imp-semantics'],
        )
        assert len(claims) == 1

        kcfg = KCFG.from_claim(self.kprove.definition, claims[0])
        proof = APRBMCProof(f'{spec_module}.{claim_id}', kcfg, proof_dir=self.proof_dir, bmc_depth=bmc_depth)
        prover = APRBMCProver(proof, is_terminal=KIMP._is_terminal, same_loop=KIMP._same_loop)
        with KCFGExplore(
            self.kprove,
            id=f'{spec_module}.{claim_id}',
        ) as kcfg_explore:
            kcfg = prover.advance_proof(
                kcfg_explore,
                # max_iterations=20,
                # execute_depth=1,
                cut_point_rules=['IMP.while'],
                # terminal_rules='IMP.while',
            )

        proof.write_proof()
        print(proof.status)

    def eq_prove(
        self,
        proof_id: str
        # max_iterations: int,
        # max_depth: int,
        # terminal_rules: Iterable[str],
        # cut_rules: Iterable[str],
        # proof_status: ProofStatus,
    ) -> None:
        # claims = self.kprove.get_claims(
        #     Path(spec_file),
        #     spec_module_name=spec_module,
        #     claim_labels=[f'{spec_module}.{claim_id}'],
        #     include_dirs=[self.haskell_dir.parent.parent.parent / 'include' / 'imp-semantics'],
        # )
        # assert len(claims) == 1

        # kcfg = KCFG.from_claim(self.kprove.definition, claims[0])
        proof = read_proof(proof_id, self.proof_dir)
        assert type(proof) == EqualityProof
        prover = EqualityProver(proof)
        with KCFGExplore(
            self.kprove,
            id=proof_id,
        ) as kcfg_explore:
            prover.advance_proof(
                kcfg_explore,
            )

        proof.write_proof()
        print(proof.status)

    def show_kcfg(
        self,
        spec_file: str,
        spec_module: str,
        claim_id: str,
        # save_directory: Path | None = None,
        # includes: Iterable[str] = (),
        # claim_labels: Iterable[str] = (),
        # exclude_claim_labels: Iterable[str] = (),
        # spec_module: str | None = None,
        # md_selector: str | None = None,
        # nodes: Iterable[str] = (),
        # node_deltas: Iterable[tuple[str, str]] = (),
        # to_module: bool = False,
        # minimize: bool = True,
        **kwargs: Any,
    ) -> None:
        proof = read_proof(f'{spec_module}.{claim_id}', proof_dir=self.proof_dir)
        if type(proof) == EqualityProof:
            raise ValueError(f'Cannot show KCFG of EqualityProof {proof.id}')
        assert type(proof) == APRProof or type(proof) == APRBMCProof
        kcfg_show = KCFGShow(self.kprove)
        res_lines = kcfg_show.show(
            proof.id,
            proof.kcfg,
            # nodes=nodes,
            # node_deltas=node_deltas,
            # to_module=to_module,
            # minimize=minimize,
            # node_printer=kevm.short_info,
        )
        print('\n'.join(res_lines))

        if hasattr(proof, '_bounded_states'):
            print('Bounded states: ')
            for state in proof._bounded_states:
                print(state)

    def kcfg_refute_node(
        self,
        spec_file: str,
        spec_module: str,
        claim_id: str,
        node_short_hash: str,
        assuming: KInner | None = None,
    ) -> None:
        proof = read_proof(f'{spec_module}.{claim_id}', proof_dir=self.proof_dir)
        assert type(proof) == APRProof
        # assert proof.status == ProofStatus.PENDING

        node_id = proof.kcfg._resolve(node_short_hash)
        node_to_refute = proof.kcfg.get_node(node_id)
        if node_to_refute is None:
            raise ValueError(f'No such node {node_short_hash}')
        refutation_id = proof.refute_node(node_to_refute, assuming=assuming)

        assert proof.proof_dir
        proof = read_proof(refutation_id, proof_dir=proof.proof_dir)
        assert type(proof) is EqualityProof
        prover = EqualityProver(proof)
        with KCFGExplore(
            self.kprove,
            id=refutation_id,
        ) as kcfg_explore:
            prover.advance_proof(kcfg_explore)

        # print('Pending subproofs: ')
        # for subproof in proof.read_subproofs():
        #     print(subproof.id)
        #     claim_id = f'{proof.id}_{source_id}_to_{target_id}'
        #     # edge_claim, subst = build_claim(
        #     #     claim_id=claim_id, init_cterm=edge.source.cterm, final_cterm=edge.target.cterm
        #     # )
        #     self.kprove._write_claim_definition(claim_id='obligation', claim=subproof.to_claim())

    def kcfg_to_dot(
        self,
        spec_file: str,
        spec_module: str,
        claim_id: str,
        # save_directory: Path | None = None,
        # includes: Iterable[str] = (),
        # claim_labels: Iterable[str] = (),
        # exclude_claim_labels: Iterable[str] = (),
        # spec_module: str | None = None,
        # md_selector: str | None = None,
        # nodes: Iterable[str] = (),
        # node_deltas: Iterable[tuple[str, str]] = (),
        # to_module: bool = False,
        # minimize: bool = True,
        **kwargs: Any,
    ) -> None:
        def node_printer(cterm: CTerm) -> Iterable[str]:
            k_cell = cterm.cells['K_CELL']
            if type(k_cell) is KSequence:
                return [self.kprove.pretty_print(k_cell.items[0])]
            else:
                return [self.kprove.pretty_print(k_cell)]

        proof = APRProof.read_proof(f'{spec_module}.{claim_id}', proof_dir=self.proof_dir)
        kcfg_show = KCFGShow(self.kprove)
        dump_dir = self.proof_dir / 'dump'
        kcfg_show.dump(
            f'{spec_module}.{claim_id}',
            proof.kcfg,
            dump_dir,
            dot=True,
            node_printer=node_printer,
            # node_printer=lambda cterm: [self.kprove.pretty_print(cterm.cells['STATE_CELL'])],
        )

    def view_kcfg(
        self,
        spec_file: str,
        spec_module: str,
        claim_id: str,
        # save_directory: Path | None = None,
        # includes: Iterable[str] = (),
        # claim_labels: Iterable[str] = (),
        # exclude_claim_labels: Iterable[str] = (),
        # spec_module: str | None = None,
        # md_selector: str | None = None,
        # nodes: Iterable[str] = (),
        # node_deltas: Iterable[tuple[str, str]] = (),
        # to_module: bool = False,
        # minimize: bool = True,
        **kwargs: Any,
    ) -> None:
        proof = read_proof(f'{spec_module}.{claim_id}', proof_dir=self.proof_dir)
        if type(proof) == APRProof or type(proof) == APRBMCProof:
            kcfg_viewer = KCFGViewer(proof.kcfg, self.kprove)
            kcfg_viewer.run()

    def show_refutation(
        self,
        node: str,
        **kwargs: Any,
    ) -> None:
        # proof = APRProof.read_proof(f'{spec_module}.{claim_id}', proof_dir=self.proof_dir)
        proof = read_proof(f'infeasible-{node}', proof_dir=self.proof_dir)

        assert type(proof) == EqualityProof
        for s in proof.pretty(self.kprove):
            print(s)
        # print(self.kprove.pretty_print(KRewrite(proof.lhs_body, proof.rhs_body)))
        # kcfg_viewer = KCFGViewer(
        #     KCFG.from_claim(defn=self.kprove.definition, claim=proof.to_claim()), kprint=self.kprove
        # )
        # kcfg_viewer.run()
