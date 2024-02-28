requires "statements.k"
// in scope via statements.k:
//   requires "expressions.k"
//   requires "variables.k"

module CALLS-SYNTAX
  imports EXPRESSIONS-SYNTAX
  imports STATEMENTS-SYNTAX
  imports ID-SYNTAX

  syntax Args ::= List{Expr, ","}

  syntax Params ::= List{Id, ","}

  // functions returning values can be called in expressions
  syntax Expr ::= Id "(" Args ")"

  syntax Stmt ::= // a function is declared using keyword ~fun~
                  "fun" Id "(" Params ")" "=" "{" Stmt "}"
                // a function can be called (for its side effects)
                | Expr ";"
                // and return a value or nothing
                | "return" Expr ";"  [strict]
                | "return" ";"
endmodule

module CALLS
  imports CALLS-SYNTAX
  imports EXPRESSIONS
  imports STATEMENTS-RULES
  imports MAP
  imports LIST
  imports BOOL

  // we have to replace the configuration from statements.k here to add cells
  configuration
    <k> $PGM:Stmt </k>
    <env> .Map </env>
    <functions> .Map </functions>
    <args> .Map </args>
    <stack> .List </stack>
    <exitcode> 0 </exitcode>

  // changed configuration, need to repeat the variable rule
  rule
    <k> X:Id => V ...</k>
    <env> X |-> V ...</env>

  // assignment rule accesses configuration
  rule
    <k> X = V:Value ; => .K ...</k>
    <env> E => E [ X <- V ] </env>

  // function declaration
  rule <k> fun F(PS) = { Body } => #declare(F, PS, Body) ... </k>

  // function call (as an expression, no ";")
  rule <k> F(AS:Args) ~> Rest => #call(F, AS, Rest) ~> .K </k>

  // returning a value
  rule return; => #return(())

  rule <k> return V; ~> _ => #return(V) </k>

  // internal functionality to call and return
  syntax KItem ::= #declare( Id , Params , Stmt )
                 | #function ( Params , Stmt )
                 | #frame ( K , Map )
                 | #call ( Id , Args , K )
                 | #bindArgs ( Args, Params )
                 | #bind ( Expr, Id )             [strict(1)]
                 | #return ( Value )

  rule <k> #declare(F, PS, Body) => .K ... </k>
       <functions> FUNS => F |-> #function(PS, Body)  FUNS </functions>
    requires notBool F in_keys(FUNS)
             // what if F has already been declared before?
             // what if there are duplicated parameters?

  rule <k> #call(F, AS, Cont) => #bindArgs(AS, PS) ~> Body ~> #return(()) </k>
       <env> E </env>
       <functions> F |-> #function(PS, Body) ... </functions>
             // what if F is not declared?
       <stack> STK => ListItem(#frame(Cont, E)) STK </stack>

  rule <k> #bindArgs(.Args, .Params) => .K ... </k>
       <args> AS => .Map </args>
       <env> _ => AS </env>

  rule <k> #bindArgs( (A , AS) , (P , PS) ) => #bind(A, P) ~> #bindArgs(AS, PS) ... </k>

  rule <k> #bind(A:Value, P:Id) => .K ...</k>
       <args> ARGS => P |-> A ARGS </args>

  rule <k> #return(V) => V ~> Cont </k>
       <stack> ListItem(#frame(Cont, Env)) Rest => Rest </stack>
       <env> _ => Env </env>

  rule <k> #return(V) => .K </k>
       <stack> .List</stack>
       <exitcode> _ => V </exitcode>

  // last item in K cell is a value: return it
  rule <k> V => #return(V) </k>

  // otherwise remove and proceed with rest
  rule <k> _:Value => .K ... </k> [owise]

  // if an expression is "called", evaluate it
  rule <k> E ; => E ... </k>

endmodule
