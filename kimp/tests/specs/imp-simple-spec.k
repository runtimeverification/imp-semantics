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

    claim [bmc-loop-concrete]: <k> while (0 <= $n) { $n = $n + -1 ; } => . ... </k>
                               <env> $n |-> (1 => ?N) </env>
      ensures 0 <Int ?N

    claim [bmc-loop-symbolic]: <k> while ($s <= $n) { $n = $n + -1 ; } => . ... </k>
                               <env> ($n |-> (1 => ?N)) ($s |-> S) </env>
      requires 0 <=Int S
       ensures 0 <Int ?N

    claim [sum-10]: <k> $s = 0; $n = 10 ; while (0 <= $n) { $s = $s + $n ; $n = $n + -1 ; } => . ... </k>
                    <env> .Map => $s |-> 55 $n |-> -1 </env>

    claim [sum-100]: <k> $s = 0; $n = 100 ; while (0 <= $n) { $s = $s + $n ; $n = $n + -1 ; } => . ... </k>
                     <env> .Map => $s |-> 5050 $n |-> -1 </env>

    claim [sum-1000]: <k> $s = 0; $n = 1000 ; while (0 <= $n) { $s = $s + $n ; $n = $n + -1 ; } => . ... </k>
                      <env> .Map => $s |-> 500500 $n |-> -1 </env>
endmodule
