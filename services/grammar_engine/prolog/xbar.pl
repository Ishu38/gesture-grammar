:- module(xbar, [parse_sentence/3, parse_cp/3, parse_tp/3, parse_vp/5, parse_dp/5]).

:- use_module(lexicon).
:- use_module(agreement).

%% ==========================================================================
%% MLAF Grammar Engine — X-bar Phrase Structure
%%
%% Implements the X-bar schema:
%%   XP -> [Spec, X']
%%   X' -> [X, Complement]
%%
%% Categories:
%%   CP (complementizer phrase / sentence)
%%   TP (tense phrase)
%%   VP (verb phrase)
%%   DP (determiner phrase / pronoun)
%%   NP (noun phrase)
%%
%% Outputs nested parse tree terms for JSON serialization.
%% ==========================================================================

%% --- Entry point: parse gesture ID list into CP tree ---
%% parse_sentence(+GestureIDs, -Tree, -Rest)

parse_sentence(GestureIDs, Tree, Rest) :-
    parse_cp(GestureIDs, Tree, Rest).

%% --- CP -> null_spec + C' -> null_C + TP (declarative main clauses) ---

parse_cp(Input, cp(null_spec, c_bar(null_c, TP)), Rest) :-
    parse_tp(Input, TP, Rest).

%% --- TP -> SubjDP + T' -> T[agr] + VP ---
%% Spec-Head agreement: T's person/number must unify with Subject DP

parse_tp(Input, tp(SubjDP, t_bar(t_head(AgrFeats), VP)), Rest) :-
    parse_dp(Input, SubjDP, nom, SubjFeats, Rest1),
    parse_vp(Rest1, VP, SubjFeats, AgrFeats, Rest).

%% --- VP -> V' -> V + ObjDP (transitive) ---

parse_vp([VerbID|Rest0], vp(v_bar(v_head(VerbID, Form, VFeats), ObjDP)), SubjFeats, AgrFeats, Rest) :-
    lex(VerbID, _, v, VFeats),
    member(transitive=yes, VFeats),
    check_agreement(SubjFeats, VerbID, VFeats, AgrResult),
    agr_result_feats(AgrResult, AgrFeats),
    agr_result_form(AgrResult, Form),
    parse_dp(Rest0, ObjDP, acc, _ObjFeats, Rest).

%% --- VP -> V' -> V (intransitive — sentence ends after verb) ---

parse_vp([VerbID|Rest], vp(v_bar(v_head(VerbID, Form, VFeats))), SubjFeats, AgrFeats, Rest) :-
    lex(VerbID, _, v, VFeats),
    check_agreement(SubjFeats, VerbID, VFeats, AgrResult),
    agr_result_feats(AgrResult, AgrFeats),
    agr_result_form(AgrResult, Form).

%% --- DP -> D' -> D (pronouns: category d) ---

parse_dp([ID|Rest], dp(d_bar(d_head(ID, PhonForm, Feats))), ExpectedCase, Feats, Rest) :-
    lex(ID, PhonForm, d, Feats),
    member(case=ExpectedCase, Feats).

%% --- DP -> null_D + NP (full nouns: category n) ---

parse_dp([ID|Rest], dp(null_d, np(n_bar(n_head(ID, PhonForm, Feats)))), ExpectedCase, AdjFeats, Rest) :-
    lex(ID, PhonForm, n, Feats),
    member(case=ExpectedCase, Feats),
    %% Nouns inherit 3rd person for agreement purposes
    AdjFeats = [person=3, number=sg, case=ExpectedCase|Feats].

%% --- Helper: extract features and form from agreement result ---

agr_result_feats(agree(_, Feats), Feats).
agr_result_feats(disagree(_, Feats), Feats).

agr_result_form(agree(Form, _), Form).
agr_result_form(disagree(Form, _), Form).
