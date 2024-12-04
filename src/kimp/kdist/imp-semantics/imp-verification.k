requires "imp.k"


module IMP-VERIFICATION-SYNTAX
    imports ID-SYNTAX

    syntax Id ::= r"\\$[a-zA-Z0-9\\_]+" [token]
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
