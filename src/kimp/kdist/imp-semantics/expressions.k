module EXPRESSIONS-SYNTAX
  imports UNSIGNED-INT-SYNTAX
  imports BOOL-SYNTAX

  syntax Value   ::= Int
                   | Bool

  syntax KResult ::= Value

  syntax Expr    ::= Value
                   | "(" Expr ")"   [bracket, format(%1%2%3)]

  syntax Expr    ::= "-" Expr       [strict, format(%1%2)]
                   | "!" Expr       [strict, format(%1%2)]
                   > left:
                     Expr "*" Expr  [seqstrict]
                   | Expr "/" Expr  [seqstrict]
                   > left:
                     Expr "+" Expr  [seqstrict]
                   | Expr "-" Expr  [seqstrict]
                   > left:
                     Expr ">=" Expr [seqstrict]
                   | Expr ">"  Expr [seqstrict]
                   | Expr "<=" Expr [seqstrict]
                   | Expr "<"  Expr [seqstrict]
                   > left:
                     Expr "==" Expr [seqstrict]
                   | Expr "!=" Expr [seqstrict]
                   > left:
                     Expr "&&" Expr [strict(1)]
                   > left:
                     Expr "||" Expr [strict(1)]
endmodule


module EXPRESSIONS-RULES
  imports EXPRESSIONS-SYNTAX
  imports INT
  imports BOOL

  rule - X => 0 -Int X
  rule ! B => notBool B

  rule X + Y => X +Int Y
  rule X - Y => X -Int Y
  rule X * Y => X *Int Y
  rule X / Y => X /Int Y

  rule I1 >= I2 => I1 >=Int I2
  rule I1 >  I2 => I1 >Int  I2
  rule I1 <= I2 => I1 <=Int I2
  rule I1 <  I2 => I1 <Int  I2

  rule B1 == B2 => B1 ==Bool B2
  rule I1 == I2 => I1 ==Int I2

  rule B1 != B2 => B1 =/=Bool B2
  rule I1 != I2 => I1 =/=Int  I2

  rule true  && B => B
  rule false && _ => false

  rule true  || _ => true
  rule false || B => B
endmodule


module EXPRESSIONS
    imports EXPRESSIONS-RULES

    configuration <k> $PGM:Expr </k>
endmodule
