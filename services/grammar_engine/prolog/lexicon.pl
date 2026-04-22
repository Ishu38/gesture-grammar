:- module(lexicon, [lex/4, theta_grid/2]).

%% ==========================================================================
%% MLAF Grammar Engine — Lexicon
%%
%% Maps each gesture from SyntacticGesture.js to Prolog facts with full
%% morphosyntactic feature bundles (person, number, case, anaphor/pronominal
%% for binding theory, theta roles for subcategorization).
%%
%% Feature encoding:
%%   lex(GestureID, PhonologicalForm, Category, FeatureList)
%%   Category: d (determiner/pronoun), v (verb), n (noun)
%%   FeatureList: Key=Value pairs for unification
%% ==========================================================================

%% --- Pronouns (D-heads, raise to Spec-TP for nominative case) ---

lex(subject_i,    'I',    d, [person=1, number=sg, case=nom, anaphor=no, pronominal=yes]).
lex(subject_you,  'You',  d, [person=2, number=sg, case=nom, anaphor=no, pronominal=yes]).
lex(subject_he,   'He',   d, [person=3, number=sg, case=nom, anaphor=no, pronominal=yes]).
lex(subject_we,   'We',   d, [person=1, number=pl, case=nom, anaphor=no, pronominal=yes]).
lex(subject_they, 'They', d, [person=3, number=pl, case=nom, anaphor=no, pronominal=yes]).
lex(subject_she,  'She',  d, [person=3, number=sg, case=nom, anaphor=no, pronominal=yes]).

%% --- Verbs (V-heads, subcategorize for DP complement) ---

lex(verb_want,  'want',  v, [tense=pres, transitive=yes, vform=base]).
lex(verb_eat,   'eat',   v, [tense=pres, transitive=yes, vform=base]).
lex(verb_see,   'see',   v, [tense=pres, transitive=yes, vform=base]).
lex(verb_grab,  'grab',  v, [tense=pres, transitive=yes, vform=base]).
lex(verb_drink, 'drink', v, [tense=pres, transitive=yes, vform=base]).
lex(verb_go,    'go',    v, [tense=pres, transitive=no,  vform=base]).
lex(verb_stop,  'stop',  v, [tense=pres, transitive=no,  vform=base]).

%% --- Nouns (N-heads, take accusative case from V) ---

lex(object_food,  'food',  n, [person=3, number=sg, case=acc, countable=yes, anaphor=no, pronominal=no]).
lex(object_water, 'water', n, [person=3, number=sg, case=acc, countable=no,  anaphor=no, pronominal=no]).
lex(object_book,  'book',  n, [person=3, number=sg, case=acc, countable=yes, anaphor=no, pronominal=no]).
lex(object_apple, 'apple', n, [person=3, number=sg, case=acc, countable=yes, anaphor=no, pronominal=no]).
lex(object_ball,  'ball',  n, [person=3, number=sg, case=acc, countable=yes, anaphor=no, pronominal=no]).
lex(object_house, 'house', n, [person=3, number=sg, case=acc, countable=yes, anaphor=no, pronominal=no]).

%% --- Theta grids (subcategorization frames) ---
%% Each verb maps to a list of thematic roles it assigns to its arguments.

theta_grid(verb_want,  [agent, theme]).
theta_grid(verb_eat,   [agent, theme]).
theta_grid(verb_see,   [experiencer, stimulus]).
theta_grid(verb_grab,  [agent, theme]).
theta_grid(verb_drink, [agent, theme]).
theta_grid(verb_go,    [agent]).
theta_grid(verb_stop,  [agent]).
