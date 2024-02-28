requires "expressions.k"

module VARIABLES-SYNTAX
  imports EXPRESSIONS-SYNTAX
  imports ID-SYNTAX

  syntax Expr ::= Id
endmodule

module VARIABLES
  imports VARIABLES-SYNTAX
  imports EXPRESSIONS
  imports MAP

  configuration
    <k>   $PGM:Expr </k>
    <env> .Map      </env>

  rule
    <k> X:Id => V ... </k>
    <env> X |-> V ... </env>
endmodule
