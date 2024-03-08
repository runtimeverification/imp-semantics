requires "imp-verification.k"

module IMP-SUM-SPEC
  imports IMP-VERIFICATION

  claim [sum-spec]:
      <k>
        while (!($n <= 0)) {
          $s = $s + $n;
          $n = $n + -1;
        }
      =>
        .K
      ...
      </k>
    <env>
        $n |-> (N:Int => 0)
        $s |-> (S:Int => S +Int ((N +Int 1) *Int N /Int 2))
    </env>
  requires N >=Int 0

  claim [two-sum-spec]:
      <k>
        while (!($n <= 0)) {
          $s1 = $s1 + $n;
          $n = $n + -1;
        }
        while (!($m <= 0)) {
          $s2 = $s2 + $m;
          $m = $m + -1;
        }
      =>
        .K
      ...
      </k>
    <env>
        $n |-> (N:Int => 0)
        $m |-> (M:Int => 0)
        $s1 |-> (S1:Int => S1 +Int ((N +Int 1) *Int N /Int 2))
        $s2 |-> (S2:Int => S2 +Int ((M +Int 1) *Int M /Int 2))
    </env>
  requires N >=Int 0
   andBool M >=Int 0

endmodule