module VALUE-SYNTAX
  imports UNSIGNED-INT-SYNTAX
  imports STRING-SYNTAX
  imports BOOL-SYNTAX

  syntax Value ::= "(" ")"
                 | Int
                 | Bool
                 | String

  syntax KResult ::= Value
endmodule

module EXPRESSIONS-SYNTAX
  imports VALUE-SYNTAX

  syntax Args     ::= List{Expr, ","}

  syntax Expr     ::= Value
                    | "(" Expr ")" [bracket]

  syntax Expr     ::= "-" Expr      [group(unary), strict, non-assoc]
                    | Expr "*" Expr [group(mul), strict, left]
                    | Expr "/" Expr [group(mul), strict, left]
                    | Expr "+" Expr [group(add), strict, left]
                    | Expr "-" Expr [group(add), strict, left]

  syntax Expr     ::= Expr "==" Expr [group(bool),   strict, non-assoc]
                    | Expr "!=" Expr [group(bool),  strict, non-assoc]
                    | Expr ">=" Expr [group(bool), strict, non-assoc]
                    | Expr ">"  Expr [group(bool),   strict, non-assoc]
                    | Expr "<=" Expr [group(bool), strict, non-assoc]
                    | Expr "<"  Expr [group(bool),   strict, non-assoc]
endmodule

module EXPRESSIONS
  imports EXPRESSIONS-SYNTAX
  imports INT
  imports BOOL

  rule   - X => 0 -Int X

  rule X + Y => X +Int Y
  rule X - Y => X -Int Y
  rule X * Y => X *Int Y
  rule X / Y => X /Int Y

  rule B1 == B2 => B1 ==Bool B2
  rule I1 == I2 => I1 ==Int I2

  rule B1 != B2 => B1 =/=Bool B2
  rule I1 != I2 => I1 =/=Int  I2

  rule I1 >= I2 => I1 >=Int I2
  rule I1 >  I2 => I1 >Int  I2
  rule I1 <= I2 => I1 <=Int I2
  rule I1 <  I2 => I1 <Int  I2
endmodule
