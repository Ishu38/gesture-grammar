:- module(binding, [check_binding/3, classify_dp/3]).

:- use_module(lexicon).

%% ==========================================================================
%% MLAF Grammar Engine — Government & Binding Theory
%%
%% Implements Principles A, B, C:
%%   Principle A: Anaphors must be bound in local domain
%%   Principle B: Pronouns must be FREE in local domain
%%   Principle C: R-expressions must be free everywhere
%%
%% Current 11-gesture vocabulary:
%%   - Pronouns (I, You, He, We, They) -> Principle B
%%   - R-expressions (food, water, book) -> Principle C
%%   - Framework extensible for reflexives (Principle A)
%% ==========================================================================

%% --- classify_dp(+GestureID, -DPType, -Principle) ---
%% Classifies a DP node by its binding type.

classify_dp(GestureID, pronoun, principle_b) :-
    lex(GestureID, _, _, Feats),
    member(anaphor=no, Feats),
    member(pronominal=yes, Feats),
    !.

classify_dp(GestureID, r_expression, principle_c) :-
    lex(GestureID, _, _, Feats),
    member(anaphor=no, Feats),
    member(pronominal=no, Feats),
    !.

classify_dp(GestureID, anaphor, principle_a) :-
    lex(GestureID, _, _, Feats),
    member(anaphor=yes, Feats),
    member(pronominal=no, Feats),
    !.

classify_dp(_, unknown, none).

%% --- check_binding(+GestureIDs, +BindingDomain, -Violations) ---
%% GestureIDs: list of gesture IDs in the sentence
%% BindingDomain: local (clause-level) or global (discourse-level)
%% Violations: list of violation(Principle, ID, Message) or empty list

check_binding(GestureIDs, local, Violations) :-
    findall(
        violation(Principle, ID, Message),
        (
            member(ID, GestureIDs),
            lex(ID, _, _, _),
            binding_violation(ID, GestureIDs, local, Principle, Message)
        ),
        Violations
    ).

check_binding(GestureIDs, global, Violations) :-
    findall(
        violation(Principle, ID, Message),
        (
            member(ID, GestureIDs),
            lex(ID, _, _, _),
            binding_violation(ID, GestureIDs, global, Principle, Message)
        ),
        Violations
    ).

%% --- binding_violation(+ID, +AllIDs, +Domain, -Principle, -Message) ---
%% Checks if a specific DP violates binding principles.

%% Principle A: Anaphor must be bound in local domain
%% (no anaphors in current vocabulary, but framework ready)
binding_violation(ID, AllIDs, local, principle_a, Message) :-
    classify_dp(ID, anaphor, principle_a),
    %% Check if there's a c-commanding antecedent in the local domain
    \+ has_local_antecedent(ID, AllIDs),
    lex(ID, Form, _, _),
    atomic_list_concat(['Anaphor ', Form, ' must be bound in its local domain'], Message).

%% Principle B: Pronoun must be FREE in local domain
%% In single-clause sentences with different referents, pronouns are fine.
%% Violation occurs if a pronoun is coindexed with a local c-commanding DP.
binding_violation(ID, AllIDs, local, principle_b, Message) :-
    classify_dp(ID, pronoun, principle_b),
    has_local_binder(ID, AllIDs),
    lex(ID, Form, _, _),
    atomic_list_concat(['Pronoun ', Form, ' must be free in its local domain'], Message).

%% Principle C: R-expression must be free everywhere
binding_violation(ID, AllIDs, _, principle_c, Message) :-
    classify_dp(ID, r_expression, principle_c),
    has_any_binder(ID, AllIDs),
    lex(ID, Form, _, _),
    atomic_list_concat(['R-expression ', Form, ' must not be bound'], Message).

%% --- Binding domain helpers ---

%% An anaphor has a local antecedent if another DP with matching features
%% appears before it (c-commands it) in the same clause.
has_local_antecedent(ID, AllIDs) :-
    lex(ID, _, _, Feats),
    member(person=P, Feats),
    member(number=N, Feats),
    member(OtherID, AllIDs),
    OtherID \= ID,
    lex(OtherID, _, _, OtherFeats),
    member(person=P, OtherFeats),
    member(number=N, OtherFeats),
    %% Must appear before (c-command approximation via linear order)
    appears_before(OtherID, ID, AllIDs).

%% A pronoun has a local binder if another DP with identical reference
%% c-commands it in the same clause (simplified: same person+number, appears before)
has_local_binder(ID, AllIDs) :-
    lex(ID, _, _, Feats),
    member(person=P, Feats),
    member(number=N, Feats),
    member(case=nom, Feats),
    %% Subject pronoun bound by another subject = violation
    member(OtherID, AllIDs),
    OtherID \= ID,
    lex(OtherID, _, _, OtherFeats),
    member(person=P, OtherFeats),
    member(number=N, OtherFeats),
    member(case=nom, OtherFeats),
    appears_before(OtherID, ID, AllIDs).

%% R-expression bound anywhere — requires explicit coreference.
%% In our vocabulary, binding violation only fires when the same gesture
%% ID appears twice (duplicate R-expression) or when a pronoun and
%% R-expression share the same referential index (future extension).
has_any_binder(ID, AllIDs) :-
    %% Same gesture ID appearing more than once signals coreference
    member(OtherID, AllIDs),
    OtherID == ID,
    appears_before(OtherID, ID, AllIDs).

%% --- Linear order helper ---

appears_before(X, Y, List) :-
    nth0(IX, List, X),
    nth0(IY, List, Y),
    IX < IY.
