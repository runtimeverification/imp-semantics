require "expressions.k"
require "variables.k"

module STATEMENTS-SYNTAX
  imports EXPRESSIONS-SYNTAX
  imports VARIABLES-SYNTAX

  syntax Stmt ::= Id "=" Expr ";"                    [strict(2)]
                | "if" "(" Expr ")" Stmt "else" Stmt [strict(1)]
                | "if" "(" Expr ")" Stmt             [strict(1), avoid]
                | "while" "(" Expr ")" Stmt          // not strict!
                // blocks of statements
                | "{" Stmt "}"
                | "{" "}"
                | Stmt Stmt [right]
endmodule

module STATEMENTS-RULES
  imports STATEMENTS-SYNTAX

  rule if ( true  ) E1 else _  => E1
  rule if ( false ) _  else E2 => E2

  rule if ( C ) E => if ( C ) E else {}

  rule
    while ( C ) E
    => if ( C ) {
        E
        while ( C ) E
       }

  rule { E } => E

  rule {} => .K

  rule S1:Stmt S2:Stmt => S1 ~> S2

endmodule

module STATEMENTS

  imports EXPRESSIONS
  imports VARIABLES-SYNTAX
  imports STATEMENTS-RULES

  configuration
    <k> $PGM:Stmt </k>   // changed!
    <env> .Map    </env>

  // changed configuration, need to repeat the variable rule
  rule
    <k> X:Id => V ...</k>
    <env> X |-> V ...</env>

  // assignment rule accesses configuration
  rule
    <k> X = V:Value ; => . ...</k>
    <env> E => E [ X <- V ] </env>
endmodule
