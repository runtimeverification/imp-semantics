requires "imp-verification.k"

module MY-SPEC
  imports IMP-VERIFICATION

  claim [trivial]: <k> 3 + 4 => 7  </k>

  claim [aClaim]:
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
