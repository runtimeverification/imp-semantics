// requires "expressions.k" in scope via variables.k
requires "variables.k"

module STATEMENTS-SYNTAX
  imports EXPRESSIONS-SYNTAX
  imports VARIABLES-SYNTAX

  syntax Stmt ::= Id "=" Expr ";"                    [strict(2), format(%1 %2 %3%4)]
                | "if" "(" Expr ")" Stmt "else" Stmt [strict(1), avoid, format(%1 %2%3%4% %5 %6 %7)] // dangling else
                | "if" "(" Expr ")" Stmt             [format(%1 %2%3%4 %5)]
                | "while" "(" Expr ")" Stmt          [format(%1 %2%3%4 %5)] // not strict!
                // blocks of statements
                | "{" Stmt "}"                       [format(%1%i%n%2%d%n%3)]
                | "{" "}"                            [format(%1%2)]
                > right:
                  Stmt Stmt                          [format(%1%n%2)]
endmodule

module STATEMENTS-RULES
  imports STATEMENTS-SYNTAX

  rule [if-true]:  if ( true  ) E1 else _  => E1
  rule [if-false]: if ( false ) _  else E2 => E2

  rule [if-else]: if ( C ) E => if ( C ) E else {}

  rule [while]:
    while ( C ) E
    =>
    if ( C ) {
      E
      while ( C ) E
    }

  rule [block]: { E } => E ~> {}

  rule [done]: {} => .K

  rule [step]: S1:Stmt S2:Stmt => S1 ~> S2

endmodule

module STATEMENTS
  imports EXPRESSIONS
  imports VARIABLES-SYNTAX
  imports STATEMENTS-RULES

  configuration
    <k   color="green">  $PGM:Stmt </k>   // changed!
    <env color="yellow"> $ENV:Map  </env>

  // changed configuration, need to repeat the variable rule
  rule [var]:
    <k> X:Id => V ...</k>
    <env> X |-> V ...</env>

  // assignment rule accesses configuration
  rule [assign]:
    <k> X = V:Value ; => .K ...</k>
    <env> E => E [ X <- V ] </env>
endmodule
