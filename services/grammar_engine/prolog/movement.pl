:- module(movement, [topicalize/3, detect_movement/2, reconstruct/2]).

:- use_module(lexicon).

%% ==========================================================================
%% MLAF Grammar Engine — Syntactic Movement & Traces
%%
%% Formal counterpart to ISLInterferenceDetector._detectSOVOrder() and
%% ._detectTopicFronting() — expressed as Prolog relations with trace
%% semantics.
%%
%% Movement types:
%%   - Topicalization: DP moves from complement-of-V to Spec-CP
%%   - V-to-T raising: V moves to T (English, not ISL)
%%   - Object shift: Object DP moves past verb
%% ==========================================================================

%% --- detect_movement(+GestureIDs, -MovementType) ---
%% Given a gesture sequence, classifies word order pattern.

detect_movement(GestureIDs, sov_order) :-
    categorize_ids(GestureIDs, Categories),
    %% SOV: subject before object before verb
    nth0(IS, Categories, subj),
    nth0(IO, Categories, obj),
    nth0(IV, Categories, verb),
    IS < IO,
    IO < IV,
    !.

detect_movement(GestureIDs, topic_fronting) :-
    categorize_ids(GestureIDs, Categories),
    %% Topic-fronting: object appears first
    nth0(0, Categories, obj),
    !.

detect_movement(GestureIDs, object_drop) :-
    categorize_ids(GestureIDs, Categories),
    %% Object drop: has subject and verb but no object
    member(subj, Categories),
    member(verb, Categories),
    \+ member(obj, Categories),
    !.

detect_movement(_, canonical_svo) :- !.

%% --- topicalize(+GestureIDs, -TopicalizedIDs, -TraceRecord) ---
%% Moves first element to front (if not already), leaves trace.

topicalize(GestureIDs, TopicalizedIDs, trace_record(topicalization, Moved, OrigIdx, 0)) :-
    categorize_ids(GestureIDs, Categories),
    %% Find the object
    nth0(OrigIdx, Categories, obj),
    OrigIdx > 0,
    nth0(OrigIdx, GestureIDs, Moved),
    %% Move object to front
    select_nth0(OrigIdx, GestureIDs, Moved, Remaining),
    TopicalizedIDs = [Moved | Remaining],
    !.

topicalize(GestureIDs, GestureIDs, trace_record(none, none, -1, -1)).

%% --- reconstruct(+GestureIDs, -ReconstructedIDs) ---
%% Undoes movement: restores fronted element to canonical SVO position.

reconstruct(GestureIDs, ReconstructedIDs) :-
    detect_movement(GestureIDs, sov_order),
    !,
    %% SOV -> SVO: move verb before object
    categorize_ids(GestureIDs, Categories),
    nth0(IS, GestureIDs, Subj), nth0(IS, Categories, subj),
    nth0(IO, GestureIDs, Obj), nth0(IO, Categories, obj),
    nth0(IV, GestureIDs, Verb), nth0(IV, Categories, verb),
    ReconstructedIDs = [Subj, Verb, Obj].

reconstruct(GestureIDs, ReconstructedIDs) :-
    detect_movement(GestureIDs, topic_fronting),
    !,
    %% OSV -> SVO: move object to end
    categorize_ids(GestureIDs, Categories),
    nth0(IO, GestureIDs, Obj), nth0(IO, Categories, obj),
    nth0(IS, GestureIDs, Subj), nth0(IS, Categories, subj),
    nth0(IV, GestureIDs, Verb), nth0(IV, Categories, verb),
    ReconstructedIDs = [Subj, Verb, Obj].

reconstruct(GestureIDs, GestureIDs) :-
    detect_movement(GestureIDs, canonical_svo),
    !.

reconstruct(GestureIDs, GestureIDs).

%% --- Internal helpers ---

%% Categorize a gesture ID into subj/verb/obj
categorize_id(ID, subj) :-
    lex(ID, _, d, Feats),
    member(case=nom, Feats), !.

categorize_id(ID, verb) :-
    lex(ID, _, v, _), !.

categorize_id(ID, obj) :-
    lex(ID, _, n, _), !.

categorize_id(_, unknown).

%% Categorize all IDs in a list
categorize_ids([], []).
categorize_ids([ID|Rest], [Cat|RestCats]) :-
    categorize_id(ID, Cat),
    categorize_ids(Rest, RestCats).

%% select_nth0(+Index, +List, -Elem, -Rest)
select_nth0(0, [H|T], H, T) :- !.
select_nth0(N, [H|T], Elem, [H|Rest]) :-
    N > 0,
    N1 is N - 1,
    select_nth0(N1, T, Elem, Rest).
