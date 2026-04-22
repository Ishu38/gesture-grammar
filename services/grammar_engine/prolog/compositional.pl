:- module(compositional, [
    compose_sentence/3,
    semantic_type/2,
    semantic_repr/2,
    lambda_apply/3
]).

:- use_module(lexicon).
:- use_module(xbar).

%% ==========================================================================
%% MLAF Grammar Engine — Compositional Semantics
%%
%% Implements the Principle of Compositionality (Frege's Principle) from
%% Partee, ter Meulen & Wall, Ch 13:
%%
%%   "The meaning of a complex expression is a function of the meanings
%%    of its parts and of the way they are syntactically combined."
%%
%% Formally: there exists a homomorphism h from the syntactic algebra
%% <E_s, F_1..F_n> to the semantic algebra <E_m, G_1..G_n> such that
%% for every syntactic rule F_i, there is a corresponding semantic rule G_i.
%%
%% Type System (Montague, PTQ):
%%   e         — entities (individuals)
%%   t         — truth values (propositions)
%%   <a, b>    — functions from type a to type b
%%
%% Semantic Representations:
%%   Pronouns:        type e, denotation = constant (i, you, he, she, we, they)
%%   Trans. Verbs:    type <e, <e, t>>, denotation = lambda x.lambda y.verb(y,x)
%%   Intrans. Verbs:  type <e, t>, denotation = lambda x.verb(x)
%%   Nouns:           type e, denotation = constant (food, water, book, ...)
%%
%% Lambda Calculus:
%%   lambda(Var, Body)  — abstraction
%%   apply(Fn, Arg)     — application (beta reduction)
%%   pred(Name, Args)   — predicate (after full reduction)
%% ==========================================================================

%% --- Semantic Types ---

semantic_type(subject_i,     e).
semantic_type(subject_you,   e).
semantic_type(subject_he,    e).
semantic_type(subject_she,   e).
semantic_type(subject_we,    e).
semantic_type(subject_they,  e).

semantic_type(verb_want,     arrow(e, arrow(e, t))).
semantic_type(verb_eat,      arrow(e, arrow(e, t))).
semantic_type(verb_see,      arrow(e, arrow(e, t))).
semantic_type(verb_grab,     arrow(e, arrow(e, t))).
semantic_type(verb_drink,    arrow(e, arrow(e, t))).
semantic_type(verb_go,       arrow(e, t)).
semantic_type(verb_stop,     arrow(e, t)).

semantic_type(object_food,   e).
semantic_type(object_water,  e).
semantic_type(object_book,   e).
semantic_type(object_apple,  e).
semantic_type(object_ball,   e).
semantic_type(object_house,  e).

%% --- Semantic Representations (Lexical Entries) ---

%% Pronouns: constant denotations
semantic_repr(subject_i,     entity(i)).
semantic_repr(subject_you,   entity(you)).
semantic_repr(subject_he,    entity(he)).
semantic_repr(subject_she,   entity(she)).
semantic_repr(subject_we,    entity(we)).
semantic_repr(subject_they,  entity(they)).

%% Transitive verbs: lambda x . lambda y . verb(y, x)
%% (object first per currying: V applied to Obj gives <e,t>, then applied to Subj gives t)
semantic_repr(verb_want,     lambda(x, lambda(y, pred(want, [y, x])))).
semantic_repr(verb_eat,      lambda(x, lambda(y, pred(eat,  [y, x])))).
semantic_repr(verb_see,      lambda(x, lambda(y, pred(see,  [y, x])))).
semantic_repr(verb_grab,     lambda(x, lambda(y, pred(grab, [y, x])))).
semantic_repr(verb_drink,    lambda(x, lambda(y, pred(drink,[y, x])))).

%% Intransitive verbs: lambda x . verb(x)
semantic_repr(verb_go,       lambda(x, pred(go,   [x]))).
semantic_repr(verb_stop,     lambda(x, pred(stop, [x]))).

%% Nouns: constant denotations
semantic_repr(object_food,   entity(food)).
semantic_repr(object_water,  entity(water)).
semantic_repr(object_book,   entity(book)).
semantic_repr(object_apple,  entity(apple)).
semantic_repr(object_ball,   entity(ball)).
semantic_repr(object_house,  entity(house)).

%% ==========================================================================
%% LAMBDA CALCULUS — Beta Reduction
%% ==========================================================================

%% lambda_apply(+Function, +Argument, -Result)
%% Beta reduction: (lambda X . Body)(Arg) = Body[X := Arg]

lambda_apply(lambda(Var, Body), Arg, Result) :-
    substitute(Var, Arg, Body, Result).

lambda_apply(Expr, _, Expr) :-
    \+ functor(Expr, lambda, 2).

%% substitute(+Var, +Value, +Expr, -Result)
%% Replace all free occurrences of Var with Value in Expr.

substitute(Var, Value, Var, Value) :- !.

substitute(Var, _Value, lambda(Var, Body), lambda(Var, Body)) :- !.
    %% Var is bound in this lambda — do not substitute inside

substitute(Var, Value, lambda(OtherVar, Body), lambda(OtherVar, NewBody)) :-
    Var \= OtherVar,
    substitute(Var, Value, Body, NewBody), !.

substitute(Var, Value, pred(Name, Args), pred(Name, NewArgs)) :-
    maplist(substitute(Var, Value), Args, NewArgs), !.

substitute(Var, Value, apply(Fn, Arg), apply(NewFn, NewArg)) :-
    substitute(Var, Value, Fn, NewFn),
    substitute(Var, Value, Arg, NewArg), !.

substitute(Var, Value, entity(X), entity(NewX)) :-
    (X == Var -> NewX = Value ; NewX = X), !.

substitute(_Var, _Value, Atom, Atom) :-
    atomic(Atom), !.

substitute(Var, Value, Term, NewTerm) :-
    compound(Term),
    Term =.. [F | Args],
    maplist(substitute(Var, Value), Args, NewArgs),
    NewTerm =.. [F | NewArgs].

%% ==========================================================================
%% COMPOSITIONAL SENTENCE INTERPRETATION
%% ==========================================================================

%% compose_sentence(+GestureIDs, -SemanticForm, -TypeResult)
%%
%% Given a list of gesture IDs, compute the compositional semantic
%% representation by:
%%   1. Looking up each lexical item's semantic representation
%%   2. Applying functional application (lambda calculus)
%%   3. Returning the fully reduced logical form
%%
%% For SVO: Subj + Verb + Obj
%%   Step 1: V(Obj) = lambda y . pred(V, [y, obj])   — type <e, t>
%%   Step 2: V(Obj)(Subj) = pred(V, [subj, obj])     — type t

%% SVO — transitive
compose_sentence([SubjID, VerbID, ObjID], SemanticForm, TypeResult) :-
    lex(SubjID, _, d, _),
    lex(VerbID, _, v, VFeats),
    member(transitive=yes, VFeats),
    lex(ObjID, _, n, _),
    !,
    semantic_repr(SubjID, SubjSem),
    semantic_repr(VerbID, VerbSem),
    semantic_repr(ObjID, ObjSem),
    %% Step 1: Apply verb to object
    lambda_apply(VerbSem, ObjSem, VerbObjSem),
    %% Step 2: Apply result to subject
    lambda_apply(VerbObjSem, SubjSem, SemanticForm),
    TypeResult = t.

%% SV — intransitive
compose_sentence([SubjID, VerbID], SemanticForm, TypeResult) :-
    lex(SubjID, _, d, _),
    lex(VerbID, _, v, VFeats),
    \+ member(transitive=yes, VFeats),
    !,
    semantic_repr(SubjID, SubjSem),
    semantic_repr(VerbID, VerbSem),
    %% Apply verb to subject
    lambda_apply(VerbSem, SubjSem, SemanticForm),
    TypeResult = t.

%% Partial: Subject only
compose_sentence([SubjID], SubjSem, e) :-
    lex(SubjID, _, d, _),
    !,
    semantic_repr(SubjID, SubjSem).

%% Partial: Subject + transitive verb (awaiting object)
compose_sentence([SubjID, VerbID], PartialSem, arrow(e, t)) :-
    lex(SubjID, _, d, _),
    lex(VerbID, _, v, VFeats),
    member(transitive=yes, VFeats),
    !,
    semantic_repr(VerbID, VerbSem),
    %% Can't fully apply yet — return partial: verb needs object
    PartialSem = awaiting_object(SubjID, VerbSem).

%% Fallback
compose_sentence(_, unknown, unknown).
