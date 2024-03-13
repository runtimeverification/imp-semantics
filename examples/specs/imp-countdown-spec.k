requires "imp-verification.k"

module IMP-COUNTDOWN-SPEC
  imports IMP-VERIFICATION

  claim [countdown-spec]:
      <k>
        while (! (0 == $s))
          $s = $s + -1;
          =>
        .K
        ...
      </k>
    <env>
        $s |-> (S => 0)
    </env>
  requires 0 <=Int S

endmodule
