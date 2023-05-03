requires "imp.k"

module IMP-VERIFICATION-SYNTAX
    syntax Id ::= "$s" [token]
                | "$s1" [token]
                | "$s2" [token]
                | "$n" [token]
                | "$m" [token]
endmodule


module IMP-VERIFICATION
    imports IMP-VERIFICATION-SYNTAX
    imports IMP

    // inequality sign normalization
    rule          A  >Int B  => B  <Int A [simplification]
    rule          A >=Int B  => B <=Int A [simplification]
    rule notBool (A  <Int B) => B <=Int A [simplification]
    rule notBool (A <=Int B) => B  <Int A [simplification]

endmodule

