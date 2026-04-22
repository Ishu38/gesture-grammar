:- module(isl_grammar, [isl_property/2, isl_parse/3, isl_word_order/1]).

:- use_module(lexicon).

%% ==========================================================================
%% MLAF Grammar Engine — ISL Typological Facts
%%
%% Declarative ISL typological profile grounded in:
%%   - Zeshan (2003): Indo-Pakistani Sign Language Grammar
%%   - Sinha (2008): A Grammar of Indian Sign Language
%%
%% ISL is an SOV language with high topic prominence, spatial agreement,
%% classifier predicates, and non-manual question formation.
%% ==========================================================================

%% --- ISL typological properties ---

isl_property(word_order, sov).
isl_property(topic_prominence, high).
isl_property(copula, omitted).
isl_property(agreement_type, spatial).
isl_property(classifier_predicates, yes).
isl_property(question_formation, non_manual).
isl_property(negation, head_shake).
isl_property(relative_clause, prenominal).
isl_property(aspect_marking, reduplication).
isl_property(pronoun_system, spatial_index).

%% --- ISL canonical word order ---

isl_word_order(sov).

%% --- ISL-specific parse rule ---
%% isl_parse(+GestureIDs, -ISL_Tree, -Status)
%% Parses SOV order into ISL-specific tree structure where V remains
%% in situ (no V-to-T raising, unlike English).

%% SOV order: Subject-Object-Verb (ISL canonical)
isl_parse([SubjID, ObjID, VerbID], isl_tp(SubjDP, isl_vp(ObjDP, VHead)), ok) :-
    lex(SubjID, SubjForm, d, SubjFeats),
    member(case=nom, SubjFeats),
    lex(ObjID, ObjForm, n, _ObjFeats),
    lex(VerbID, VerbForm, v, _VerbFeats),
    SubjDP = isl_dp(SubjID, SubjForm, SubjFeats),
    ObjDP = isl_dp(ObjID, ObjForm, [case=acc]),
    VHead = isl_v_head(VerbID, VerbForm).

%% SV order: Subject-Verb (intransitive or object-drop, common in ISL)
isl_parse([SubjID, VerbID], isl_tp(SubjDP, isl_vp(VHead)), ok) :-
    lex(SubjID, SubjForm, d, SubjFeats),
    member(case=nom, SubjFeats),
    lex(VerbID, VerbForm, v, _VerbFeats),
    SubjDP = isl_dp(SubjID, SubjForm, SubjFeats),
    VHead = isl_v_head(VerbID, VerbForm).

%% Topic-fronted: Object-Subject-Verb (ISL topic prominence)
isl_parse([ObjID, SubjID, VerbID], isl_cp(TopicDP, isl_tp(SubjDP, isl_vp(trace(obj), VHead))), ok) :-
    lex(ObjID, ObjForm, n, _ObjFeats),
    lex(SubjID, SubjForm, d, SubjFeats),
    member(case=nom, SubjFeats),
    lex(VerbID, VerbForm, v, _VerbFeats),
    TopicDP = isl_topic(ObjID, ObjForm),
    SubjDP = isl_dp(SubjID, SubjForm, SubjFeats),
    VHead = isl_v_head(VerbID, VerbForm).

%% Fallback: unrecognized order
isl_parse(GestureIDs, isl_unknown(GestureIDs), parse_error) :-
    length(GestureIDs, L),
    L > 0.

isl_parse([], isl_empty, empty).
