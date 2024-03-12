requires "imp-verification.k"

module IMP-IFS-SPEC
  imports IMP-VERIFICATION

  claim [assignment]:
      <k>
        $n = N;
      =>
        .K
      ...
      </k>
    <env>
        $n |-> (_ => ?N')
    </env>
    ensures ?N' ==Int N

  claim [branching-if]:
      <k>
        $n = N:Int;
        $m = M:Int;
        if ($n == $m) { $k = 1; } else { $k = 0; }
      =>
        .K
      ...
      </k>
    <env>
        $n |-> (_:Int => ?_:Int)
        $m |-> (_:Int => ?_:Int)
        $k |-> (_:Int => ?K:Int)
    </env>
    requires isUInt64(N) andBool isUInt64(M)
    ensures ?K ==Int 1 orBool ?K ==Int 0

endmodule