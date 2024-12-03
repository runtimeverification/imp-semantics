requires "expr.k"


module CALC-SYNTAX
  imports EXPR-SYNTAX
  imports ID-SYNTAX

  syntax Expr ::= Id

  syntax Stmt ::= Id "=" Expr ";" [group(stmt), strict(2), format(%1 %2 %3%4)]
  syntax Stmt ::= right:
                  Stmt Stmt       [group(seq-stmt), format(%1%n%2)]

  syntax priority stmt > seq-stmt
endmodule


module CALC
  imports CALC-SYNTAX
  imports EXPR-RULES

  configuration
    <k   color="green">  $PGM:Stmt </k>
    <env color="yellow"> $ENV:Map  </env>

  rule [step]: <k> S1:Stmt S2:Stmt => S1 ~> S2 ... </k>

  rule [var]:
    <k> X:Id => V ... </k>
    <env> X |-> V ... </env>

  rule [assign]:
    <k> X = V:Value ; => .K ... </k>
    <env> E => E [ X <- V ] </env>
endmodule
