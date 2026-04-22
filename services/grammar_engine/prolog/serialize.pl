:- module(serialize, [
    serialize_tree/2,
    serialize_agreement/5,
    serialize_validate/3,
    serialize_interference/2,
    serialize_transform/4
]).

:- use_module(lexicon).
:- use_module(agreement).
:- use_module(xbar).
:- use_module(binding).
:- use_module(subcategorization).
:- use_module(contrastive).
:- use_module(movement).

%% ==========================================================================
%% Serialization helpers — convert compound Prolog terms into nested lists
%% that pyswip can marshal cleanly to Python (atoms + lists only).
%% ==========================================================================

%% --- serialize_tree(+GestureIDs, -TreeList) ---
%% Parses and serializes the X-bar tree as nested lists.
%% Each node: [Label, Children...] where Children are also nested lists.

serialize_tree(GestureIDs, TreeList) :-
    parse_sentence(GestureIDs, Tree, []),
    !,
    term_to_list(Tree, TreeList).

serialize_tree(_, []).

%% --- term_to_list: recursive compound term → nested list ---

term_to_list(Term, [Functor|ArgLists]) :-
    compound(Term),
    Term =.. [Functor|Args],
    maplist(term_to_list, Args, ArgLists),
    !.

term_to_list(Atom, Atom) :-
    atom(Atom), !.

term_to_list(Num, Num) :-
    number(Num), !.

term_to_list(List, Serialized) :-
    is_list(List),
    !,
    maplist(term_to_list, List, Serialized).

term_to_list(Other, Other).

%% --- serialize_agreement(+SubjID, +VerbID, -Agrees, -InflectedForm, -AgrFeatsFlat) ---

serialize_agreement(SubjID, VerbID, Agrees, InflectedForm, AgrFeatsFlat) :-
    lex(SubjID, _, _, SF),
    lex(VerbID, _, _, VF),
    check_agreement(SF, VerbID, VF, Result),
    !,
    (Result = agree(InflectedForm, AgrFeats) ->
        Agrees = yes,
        flatten_feats(AgrFeats, AgrFeatsFlat)
    ;
        Result = disagree(InflectedForm, _),
        Agrees = no,
        AgrFeatsFlat = []
    ).

serialize_agreement(_, _, no, unknown, []).

%% --- flatten_feats: [k=v, ...] → [k, v, k, v, ...] ---

flatten_feats([], []).
flatten_feats([K=V|Rest], [K, V|FlatRest]) :-
    flatten_feats(Rest, FlatRest).
flatten_feats([_|Rest], FlatRest) :-
    flatten_feats(Rest, FlatRest).

%% --- serialize_validate(+GestureIDs, -Grammatical, -ResultPairs) ---
%% Returns flattened key-value pairs for the validation result.

serialize_validate(GestureIDs, Grammatical, ResultPairs) :-
    %% 1. Try parse
    (parse_sentence(GestureIDs, _, []) ->
        Grammatical = yes, ParseOk = yes
    ;
        Grammatical = no, ParseOk = no
    ),

    %% 2. Theta check
    find_verb_id(GestureIDs, VerbID),
    (VerbID \= none ->
        exclude_id(GestureIDs, VerbID, ArgIDs),
        check_theta_criterion(VerbID, ArgIDs, ThetaResult),
        (ThetaResult = satisfied -> ThetaSat = yes ; ThetaSat = no)
    ;
        ThetaSat = unknown, _ThetaResult = none
    ),

    %% 3. Binding check
    check_binding(GestureIDs, local, BindingViolations),
    length(BindingViolations, NBindViol),

    ResultPairs = [parse_ok, ParseOk, theta_satisfied, ThetaSat, binding_violations, NBindViol].

%% --- serialize_interference(+GestureIDs, -PatternsList) ---
%% Each pattern: [Type, Severity, Description]

serialize_interference(GestureIDs, PatternsList) :-
    detect_interference(GestureIDs, Interferences),
    maplist(interference_to_list, Interferences, PatternsList).

interference_to_list(interference(Type, Severity, Desc), [Type, Severity, Desc]).

%% --- serialize_transform(+GestureIDs, -EnglishOrder, -TransformType, -OpsList) ---

serialize_transform(GestureIDs, EnglishOrder, TransformType, OpsList) :-
    transform_isl_to_english(GestureIDs, EnglishOrder, Record),
    (Record = transform(TransformType, Ops) ->
        maplist(op_to_list, Ops, OpsList)
    ;
        TransformType = unknown, OpsList = []
    ).

op_to_list(operation(Name, Desc), [Name, Desc]).

%% --- Helpers ---

find_verb_id([], none).
find_verb_id([ID|_], ID) :- lex(ID, _, v, _), !.
find_verb_id([_|Rest], VID) :- find_verb_id(Rest, VID).

exclude_id([], _, []).
exclude_id([ID|Rest], ID, Result) :- !, exclude_id(Rest, ID, Result).
exclude_id([ID|Rest], ExID, [ID|Result]) :- exclude_id(Rest, ExID, Result).
