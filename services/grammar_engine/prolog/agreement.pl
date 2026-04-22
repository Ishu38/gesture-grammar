:- module(agreement, [check_agreement/4, inflect_verb/5, unify_features/3]).

:- use_module(lexicon).

%% ==========================================================================
%% MLAF Grammar Engine — Feature Unification & Agreement
%%
%% Uses Prolog's native unification for morphosyntactic agreement.
%% English subject-verb agreement: 3sg present adds -s.
%% ==========================================================================

%% --- check_agreement(+SubjFeats, +VerbID, +VerbFeats, -Result) ---
%% Returns agree(InflectedForm, MergedFeats) or disagree(BaseForm, Reason)

check_agreement(SubjFeats, VerbID, VerbFeats, agree(InflectedForm, MergedFeats)) :-
    member(person=Person, SubjFeats),
    member(number=Number, SubjFeats),
    inflect_verb(Person, Number, pres, VerbID, InflectedForm),
    merge_agr_features(SubjFeats, VerbFeats, MergedFeats),
    !.

check_agreement(SubjFeats, VerbID, VerbFeats, disagree(BaseForm, mismatch(SubjFeats, VerbFeats))) :-
    lex(VerbID, BaseForm, v, _).

%% --- inflect_verb(+Person, +Number, +Tense, +VerbID, -InflectedForm) ---
%% English morphology: 3sg present adds -s

inflect_verb(3, sg, pres, VerbID, InflectedForm) :-
    !,
    lex(VerbID, BaseForm, v, _),
    atom_concat(BaseForm, 's', InflectedForm).

inflect_verb(_, _, pres, VerbID, BaseForm) :-
    lex(VerbID, BaseForm, v, _).

inflect_verb(_, _, Tense, VerbID, BaseForm) :-
    Tense \= pres,
    lex(VerbID, BaseForm, v, _),
    !.

%% --- unify_features(+F1, +F2, -Merged) ---
%% Merge two feature lists. Fails on conflicting values (Prolog backtracking).

unify_features([], F2, F2).
unify_features([K=V|Rest], F2, Merged) :-
    (member(K=V2, F2) ->
        (V = V2 ->
            %% Same key, same value: keep one copy
            unify_features(Rest, F2, Merged)
        ;
            %% Conflict: fail
            fail
        )
    ;
        %% Key not in F2: add it
        unify_features(Rest, [K=V|F2], Merged)
    ).

%% --- merge_agr_features: non-failing merge for agreement result ---

merge_agr_features(SubjFeats, VerbFeats, Merged) :-
    (unify_features(SubjFeats, VerbFeats, M) ->
        Merged = M
    ;
        append(SubjFeats, VerbFeats, Merged)
    ).
