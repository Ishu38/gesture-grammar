:- module(chomsky_hierarchy, [
    grammar_class/2,
    rule_type/2,
    grammar_capabilities/1,
    chomsky_level/2
]).

%% ==========================================================================
%% MLAF Grammar Engine — Chomsky Hierarchy Classification
%%
%% Classifies MLAF's grammar rules by their position in the Chomsky Hierarchy
%% (Partee, ter Meulen & Wall, Ch 16):
%%
%%   Type 3 (Regular):         A → aB | a
%%     - Gesture sequence validation (finite state)
%%     - Gesture lifecycle DFA
%%
%%   Type 2 (Context-Free):    A → gamma (gamma in (V u T)*)
%%     - X-bar phrase structure rules
%%     - Earley parser
%%
%%   Type 1 (Context-Sensitive): aAb → a-gamma-b (|lhs| <= |rhs|)
%%     - Subject-verb agreement (person/number feature checking)
%%     - Binding theory (Principles A, B, C — structural conditions)
%%     - Theta criterion (argument structure constraints)
%%
%%   Type 0 (Unrestricted):    alpha → beta (no restrictions)
%%     - Not used in MLAF (would lose decidability)
%%
%% Natural languages are MILDLY CONTEXT-SENSITIVE (Joshi 1985):
%%   - Beyond CFG (cross-serial dependencies)
%%   - Below full CSG (polynomial parsing)
%%   - MLAF handles this via CFG base + context-sensitive feature checking
%%
%% This module provides introspective queries: the grammar engine can
%% report its own formal power, which is critical for:
%%   1. Patent documentation (proves the theoretical grounding)
%%   2. Academic reviewers (shows awareness of formal limits)
%%   3. Runtime: knowing which parser to dispatch for which check
%% ==========================================================================

%% --- Grammar Component Classification ---

%% grammar_class(+Component, -ChomskyType)
%% Maps each MLAF grammar component to its Chomsky Hierarchy level.

grammar_class(gesture_lifecycle_dfa,    type_3_regular).
grammar_class(gesture_sequence_fsm,     type_3_regular).
grammar_class(tense_zone_classifier,    type_3_regular).

grammar_class(xbar_phrase_structure,    type_2_context_free).
grammar_class(earley_parser,            type_2_context_free).
grammar_class(dp_np_vp_rules,           type_2_context_free).

grammar_class(subject_verb_agreement,   type_1_context_sensitive).
grammar_class(binding_theory,           type_1_context_sensitive).
grammar_class(theta_criterion,          type_1_context_sensitive).
grammar_class(case_assignment,          type_1_context_sensitive).
grammar_class(isl_interference,         type_1_context_sensitive).

grammar_class(compositional_semantics,  type_1_context_sensitive).

%% --- Rule Type Classification ---

%% rule_type(+RuleName, -Description)

rule_type(type_3_regular, rule_desc(
    'Type 3 — Regular Grammar',
    'Recognized by finite automata (DFA/NFA). O(n) parsing.',
    'Gesture sequences, tense zones, lifecycle state machine'
)).

rule_type(type_2_context_free, rule_desc(
    'Type 2 — Context-Free Grammar',
    'Recognized by pushdown automata. O(n^3) Earley parsing.',
    'X-bar phrase structure: CP → TP → VP → DP/NP'
)).

rule_type(type_1_context_sensitive, rule_desc(
    'Type 1 — Context-Sensitive Grammar',
    'Feature unification across non-local domains. Polynomial complexity.',
    'Agreement, binding, theta roles, ISL transfer detection'
)).

%% --- Chomsky Level Numeric ---

chomsky_level(type_3_regular, 3).
chomsky_level(type_2_context_free, 2).
chomsky_level(type_1_context_sensitive, 1).
chomsky_level(type_0_unrestricted, 0).

%% --- Grammar Capabilities Report ---

%% grammar_capabilities(-Report)
%% Returns a comprehensive report of MLAF's formal grammar capabilities.

grammar_capabilities(Report) :-
    findall(
        [Component, TypeAtom, Level],
        (grammar_class(Component, TypeAtom), chomsky_level(TypeAtom, Level)),
        Components
    ),
    length(Components, Total),
    findall(C, (member([C, type_3_regular, _], Components)), RegComps),
    findall(C, (member([C, type_2_context_free, _], Components)), CFComps),
    findall(C, (member([C, type_1_context_sensitive, _], Components)), CSComps),
    length(RegComps, NReg),
    length(CFComps, NCF),
    length(CSComps, NCS),
    Report = capabilities(
        total_components(Total),
        type_3_regular(NReg, RegComps),
        type_2_context_free(NCF, CFComps),
        type_1_context_sensitive(NCS, CSComps),
        overall_power(mildly_context_sensitive),
        note('Natural languages are mildly context-sensitive (Joshi 1985). MLAF uses CFG base + CS feature checking.')
    ).
