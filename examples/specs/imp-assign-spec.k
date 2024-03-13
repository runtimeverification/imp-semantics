requires "imp-verification.k"

module IMP-ASSIGN-SPEC
  imports IMP-VERIFICATION

  claim [countdown-spec]:
      <k>
        while (! (0 == $s))
          $s = 0;
        =>
        .K
        ...
      </k>
    <env>
        $s |-> (_:Int => 0)
    </env>

endmodule
