requires "imp-verification.k"

module IMP-SIMPLE-SPEC
    imports IMP-VERIFICATION

    claim [addition-1]: <k> 3 + 4 => 7 ... </k>

    claim [addition-2]: <k> $s + $n => 7 ... </k> <env> $s |-> 3 $n |-> 4 </env>

    claim [addition-var]: <k> 3 + N => 7 ... </k> requires N ==Int 4

    claim [pre-branch-proved]: <k> $n = 7 ; if (B:Bool) { $n = 6 ; } else { $n = 5 ; } => if (B:Bool) { $n = 6 ; } else { $n = 5 ; } </k>
                               <env> $n |-> (0 => 7) </env>

    claim [branching]: <k> $n = 7 ; if (_B:Bool) { $n = 6 ; } else { $n = 5 ; } => . ... </k>
                       <env> $n |-> (0 => ?N) </env>
                       ensures ?N ==Int 5 orBool ?N ==Int 6

    claim [branching-program]: <k> $n = 7 ; if (_B:Bool) { $n = 6 ; } else { $n = 5 ; } => . ... </k>
                               <env> _ => ?_ </env>

    claim [branching-deadcode]: <k> $n = 7 ; if (B:Bool) { $n = 6 ; } else { if (! B:Bool) { $n = 5 ; } else { $n = 4 ; } } => . ... </k>
                                <env> $n |-> (0 => ?N) </env>
                                ensures ?N ==Int 5 orBool ?N ==Int 6

    claim [while-cut-rule]: <k> while (B:Bool) BB:Stmt => if (B) { BB while (B) BB } else {} ... </k>

    claim [while-cut-rule-delayed]: <k> $n = 7 ; while (B:Bool) BB:Stmt => if (B) { BB while (B) BB } else {} ... </k>
                                    <env> $n |-> (_ => ?_) </env>

    claim [sum-loop]:
        <k> while (0 < $n) { $s = $s + $n ; $n = $n - 1 ; } => . ... </k>
        <env> $s |-> S $n |-> N => $s |-> S +Int (N *Int (N +Int 1)) /Int 2 $n |-> 0 </env>
      requires 0 <=Int N // Try without this!

    claim [sum-N]:
        <k> $s = 0; $n = N; while (0 < $n) { $s = $s + $n ; $n = $n - 1 ; } => . ... </k>
        <env> .Map => $s |-> (N *Int (N +Int 1)) /Int 2 $n |-> 0 </env>
      requires 0 <=Int N

    claim [bmc-loop-concrete]: <k> while (0 <= $n) { $n = $n + -1 ; } => . ... </k>
                               <env> $n |-> (1 => ?N) </env>
      ensures 0 <Int ?N

    claim [bmc-loop-symbolic]: <k> while ($s <= $n) { $n = $n + -1 ; } => . ... </k>
                               <env> ($n |-> (1 => ?N)) ($s |-> S) </env>
      requires 0 <=Int S
       ensures 0 <Int ?N

endmodule
