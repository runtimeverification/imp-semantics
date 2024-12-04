requires "calc.k"


module IMP-SYNTAX
  imports CALC-SYNTAX

  syntax Stmt ::= "if" "(" Expr ")" Stmt "else" Stmt [group(stmt), strict(1), avoid, format(%1 %2%3%4% %5 %6 %7)] // dangling else
                | "if" "(" Expr ")" Stmt             [group(stmt), format(%1 %2%3%4 %5)]
                | "while" "(" Expr ")" Stmt          [group(stmt), format(%1 %2%3%4 %5)] // not strict!
                | "{" Stmt "}"                       [group(stmt), format(%1%i%n%2%d%n%3)]
                | "{" "}"                            [group(stmt), format(%1%2)]
endmodule


module IMP
  imports IMP-SYNTAX
  imports CALC

  rule [if-true]:  <k> if ( true  ) S1 else _  => S1 ... </k>
  rule [if-false]: <k> if ( false ) _  else S2 => S2 ... </k>

  rule [if-else]: <k> if ( C ) S => if ( C ) S else {} ... </k>

  rule [while]:
    <k>
      while ( C ) S
      =>
      if ( C ) {
        S
        while ( C ) S
      }
      ...
    </k>

  rule [block]: <k> { S } => S ~> { } ... </k>

  rule [done]: <k> { } => .K ... </k>
endmodule
