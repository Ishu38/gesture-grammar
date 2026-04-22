:- module(contrastive, [transform_isl_to_english/3, detect_interference/2]).

:- use_module(lexicon).
:- use_module(movement).

%% ==========================================================================
%% MLAF Grammar Engine — ISL<->English Contrastive Analysis
%%
%% Formal replacement for ISLInterferenceDetector.js.
%% Each transform carries movement operation records with formal linguistic
%% labels (V-raising, de-topicalization, object shift) rather than ad-hoc
%% string descriptions.
%% ==========================================================================

%% --- transform_isl_to_english(+GestureIDs, -EnglishOrder, -TransformRecord) ---

%% Case 1: SOV -> SVO (V-raising + object shift)
transform_isl_to_english(GestureIDs, EnglishOrder, transform(sov_to_svo, Operations)) :-
    detect_movement(GestureIDs, sov_order),
    !,
    categorize_and_extract(GestureIDs, Subj, Verb, Obj),
    EnglishOrder = [Subj, Verb, Obj],
    Operations = [
        operation(v_raising, 'V moves from clause-final to T position'),
        operation(object_shift, 'Object DP moves to complement-of-V from pre-verbal position')
    ].

%% Case 2: OSV (topic-fronted) -> SVO (de-topicalization + subject to Spec-TP)
transform_isl_to_english(GestureIDs, EnglishOrder, transform(osv_to_svo, Operations)) :-
    detect_movement(GestureIDs, topic_fronting),
    !,
    GestureIDs = [Obj, Subj, Verb],
    EnglishOrder = [Subj, Verb, Obj],
    Operations = [
        operation(de_topicalization, 'Topic DP moves from Spec-CP back to complement-of-V'),
        operation(subject_raising, 'Subject DP raises to Spec-TP')
    ].

%% Case 3: SV (object drop) -> SVO + missing_object marker
transform_isl_to_english(GestureIDs, EnglishOrder, transform(object_drop_repair, Operations)) :-
    detect_movement(GestureIDs, object_drop),
    !,
    GestureIDs = [Subj, Verb | _Rest],
    EnglishOrder = [Subj, Verb, missing_object],
    Operations = [
        operation(object_insertion, 'English requires overt object for transitive verbs'),
        operation(theta_violation, 'Theme theta-role unassigned (Theta Criterion violation)')
    ].

%% Case 4: SVO -> SVO (no transform needed, canonical English)
transform_isl_to_english(GestureIDs, GestureIDs, transform(none, [])) :-
    detect_movement(GestureIDs, canonical_svo),
    !.

%% Fallback
transform_isl_to_english(GestureIDs, GestureIDs, transform(unknown, [])).

%% --- detect_interference(+GestureIDs, -Interferences) ---
%% Uses findall/3 over all interference patterns.

detect_interference(GestureIDs, Interferences) :-
    findall(
        interference(Type, Severity, Description),
        interference_pattern(GestureIDs, Type, Severity, Description),
        Interferences
    ).

%% --- Interference patterns ---

%% SOV word order (ISL canonical, English error)
interference_pattern(GestureIDs, sov_order, error, Description) :-
    detect_movement(GestureIDs, sov_order),
    categorize_and_extract(GestureIDs, Subj, Verb, Obj),
    lex(Subj, SF, _, _), lex(Verb, VF, _, _), lex(Obj, OF, _, _),
    atomic_list_concat([
        'SOV word order detected: "', SF, ' ', OF, ' ', VF,
        '". ISL uses SOV (Subject-Object-Verb) while English requires SVO. ',
        'Apply V-to-T raising to move verb before object.'
    ], Description).

%% Topic fronting (ISL high topic prominence)
interference_pattern(GestureIDs, topic_fronting, warning, Description) :-
    detect_movement(GestureIDs, topic_fronting),
    GestureIDs = [Obj | _],
    lex(Obj, OF, _, _),
    atomic_list_concat([
        'Topic-fronting detected: "', OF,
        '" moved to sentence-initial position. ',
        'ISL has high topic prominence; English uses SVO order. ',
        'Move topic back to complement-of-V position.'
    ], Description).

%% Object drop (ISL allows, English requires overt objects for transitives)
interference_pattern(GestureIDs, object_drop, warning, Description) :-
    detect_movement(GestureIDs, object_drop),
    GestureIDs = [_Subj, Verb | _],
    lex(Verb, VF, v, VFeats),
    member(transitive=yes, VFeats),
    atomic_list_concat([
        'Object drop detected after transitive verb "', VF,
        '". ISL allows pro-drop of objects; English requires overt ',
        'object for transitive verbs (Theta Criterion).'
    ], Description).

%% --- Helper: extract subject, verb, object from categorized sequence ---

categorize_and_extract(GestureIDs, Subj, Verb, Obj) :-
    member(Subj, GestureIDs), lex(Subj, _, d, Feats), member(case=nom, Feats),
    member(Verb, GestureIDs), lex(Verb, _, v, _),
    member(Obj, GestureIDs), lex(Obj, _, n, _),
    !.
