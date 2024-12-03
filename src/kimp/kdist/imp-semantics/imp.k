requires "calc.k"


module IMP-SYNTAX
  imports CALC-SYNTAX

  syntax Stmt ::= "if" "(" Expr ")" Stmt "else" Stmt [group(stmt), strict(1), avoid, format(%1 %2%3%4% %5 %6 %7)] // dangling else
                | "if" "(" Expr ")" Stmt             [group(stmt), format(%1 %2%3%4 %5)]
                | "while" "(" Expr ")" Stmt          [group(stmt), format(%1 %2%3%4 %5)] // not strict!
                | "{" Stmt "}"                       [group(stmt), format(%1%i%n%2%d%n%3)]
                | "{" "}"                            [group(stmt), format(%1%2)]
endmodule


module IMP-RULES
  imports IMP-SYNTAX

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
endmodule


module IMP
  imports IMP-RULES
  imports CALC
endmodule
