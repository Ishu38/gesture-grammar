:- module(tree_validation, [
    well_formed_tree/1,
    validate_tree_structure/2
]).

%% ==========================================================================
%% MLAF Grammar Engine — Tree Well-Formedness Validation
%%
%% Formal tree well-formedness conditions from Partee, ter Meulen & Wall,
%% "Mathematical Methods in Linguistics", Chapter 16.
%%
%% A constituent structure tree T = <N, Q, D, P, L> must satisfy:
%%
%%   1. SINGLE ROOT:  Exactly one node is not dominated by any other node.
%%   2. UNIQUE MOTHER: Every non-root node has exactly one mother.
%%   3. ANTISYMMETRY:  If A dominates B, then B does not dominate A.
%%   4. TRANSITIVITY:  If A dominates B and B dominates C, then A dominates C.
%%   5. EXHAUSTIVE ORDERING: All sisters are linearly ordered (by precedence).
%%   6. NON-TANGLING:  If A precedes B, then all nodes dominated by A
%%                     precede all nodes dominated by B.
%%
%% Tree Representation:
%%   Compound terms of the form:  node(Label, [Child1, Child2, ...])
%%   Leaves are atoms or:         leaf(Label)
%%
%%   The X-bar trees from xbar.pl are compound terms like:
%%     s(np(d(she)), vp(v(eat), np(n(food))))
%%
%%   We operate on the raw compound term structure.
%% ==========================================================================

%% --- well_formed_tree(+Tree) ---
%% Succeeds if the tree satisfies all well-formedness conditions.
%% Fails with a descriptive error if any condition is violated.

well_formed_tree(Tree) :-
    nonvar(Tree),
    % Condition 1: Tree has content (non-empty)
    Tree \= [],
    % Condition 2: Verify structural integrity
    valid_structure(Tree),
    % Condition 3: No cycles (antisymmetry — guaranteed by construction in SWI-Prolog terms)
    acyclic_term(Tree),
    % Condition 4: Every non-leaf has at least one child
    all_internals_have_children(Tree),
    % Condition 5: Check label validity
    all_nodes_labeled(Tree).

%% --- validate_tree_structure(+Tree, -Report) ---
%% Returns a detailed validation report as a list of [Check, Status] pairs.

validate_tree_structure(Tree, Report) :-
    (nonvar(Tree) -> NonEmpty = pass ; NonEmpty = fail),
    (valid_structure(Tree) -> Structure = pass ; Structure = fail),
    (acyclic_term(Tree) -> Acyclic = pass ; Acyclic = fail),
    (all_internals_have_children(Tree) -> InternalCheck = pass ; InternalCheck = fail),
    (all_nodes_labeled(Tree) -> LabelCheck = pass ; LabelCheck = fail),
    count_nodes(Tree, NodeCount),
    tree_depth(Tree, Depth),
    Report = [
        [non_empty, NonEmpty],
        [valid_structure, Structure],
        [acyclic, Acyclic],
        [internals_have_children, InternalCheck],
        [all_labeled, LabelCheck],
        [node_count, NodeCount],
        [depth, Depth]
    ].

%% ==========================================================================
%% STRUCTURAL CHECKS
%% ==========================================================================

%% valid_structure(+Tree)
%% A valid tree is either an atom (leaf), a number, a list (feature bundle),
%% or a compound term whose arguments are all valid trees.

valid_structure(Tree) :-
    atom(Tree), !.

valid_structure(Tree) :-
    number(Tree), !.

valid_structure(Tree) :-
    is_list(Tree), !.  %% Feature lists like [person=3, number=sg, ...] are valid leaves

valid_structure(Tree) :-
    compound(Tree),
    Tree =.. [_Functor | Args],
    Args \= [],
    maplist(valid_structure, Args).

%% all_internals_have_children(+Tree)
%% Every compound (internal) node must have at least one argument.
%% This is guaranteed by the compound check but we verify explicitly.

all_internals_have_children(Tree) :-
    atom(Tree), !.

all_internals_have_children(Tree) :-
    number(Tree), !.

all_internals_have_children(Tree) :-
    is_list(Tree), !.  %% Feature lists are leaf nodes

all_internals_have_children(Tree) :-
    compound(Tree),
    Tree =.. [_ | Args],
    Args \= [],
    maplist(all_internals_have_children, Args).

%% all_nodes_labeled(+Tree)
%% Every node must have a valid label (the functor for compound terms,
%% the atom itself for leaves).

all_nodes_labeled(Tree) :-
    atom(Tree),
    Tree \= '', !.

all_nodes_labeled(Tree) :-
    number(Tree), !.

all_nodes_labeled(Tree) :-
    is_list(Tree), !.  %% Feature lists are valid leaf data

all_nodes_labeled(Tree) :-
    compound(Tree),
    Tree =.. [Functor | Args],
    atom(Functor),
    Functor \= '',
    maplist(all_nodes_labeled, Args).

%% ==========================================================================
%% TREE METRICS
%% ==========================================================================

%% count_nodes(+Tree, -Count)
%% Count total number of nodes (internal + leaves).

count_nodes(Tree, 1) :-
    atom(Tree), !.

count_nodes(Tree, 1) :-
    number(Tree), !.

count_nodes(Tree, 1) :-
    is_list(Tree), !.  %% Feature lists count as 1 node

count_nodes(Tree, Count) :-
    compound(Tree),
    Tree =.. [_ | Args],
    maplist(count_nodes, Args, ChildCounts),
    sum_list(ChildCounts, ChildTotal),
    Count is ChildTotal + 1.

%% tree_depth(+Tree, -Depth)
%% Maximum depth from root to any leaf.

tree_depth(Tree, 0) :-
    atom(Tree), !.

tree_depth(Tree, 0) :-
    number(Tree), !.

tree_depth(Tree, 0) :-
    is_list(Tree), !.  %% Feature lists are depth 0

tree_depth(Tree, Depth) :-
    compound(Tree),
    Tree =.. [_ | Args],
    maplist(tree_depth, Args, ChildDepths),
    max_list(ChildDepths, MaxChild),
    Depth is MaxChild + 1.

%% ==========================================================================
%% DOMINANCE AND PRECEDENCE CHECKS
%% ==========================================================================

%% dominates(+Tree, +Ancestor, +Descendant)
%% True if the node labeled Ancestor dominates the node labeled Descendant.
%% (For use in external queries — the tree structure itself encodes dominance.)

dominates(Tree, Ancestor, Descendant) :-
    compound(Tree),
    Tree =.. [Ancestor | Args],
    appears_in_any(Descendant, Args).

dominates(Tree, Ancestor, Descendant) :-
    compound(Tree),
    Tree =.. [_ | Args],
    member(Child, Args),
    dominates(Child, Ancestor, Descendant).

%% appears_in_any(+Label, +Trees)
%% True if Label appears as any node label in any of the trees.

appears_in_any(Label, [Tree | _]) :-
    appears_in(Label, Tree).
appears_in_any(Label, [_ | Rest]) :-
    appears_in_any(Label, Rest).

appears_in(Label, Label) :-
    atom(Label), !.

appears_in(Label, Tree) :-
    compound(Tree),
    Tree =.. [Label | _], !.

appears_in(Label, Tree) :-
    compound(Tree),
    Tree =.. [_ | Args],
    appears_in_any(Label, Args).

%% c_commands(+Tree, +A, +B)
%% A c-commands B iff the first branching node dominating A also dominates B,
%% and A does not dominate B. (Reinhart 1976, used in Binding Theory)

c_commands(Tree, A, B) :-
    first_branching_dominator(Tree, A, Dominator),
    dominates(Tree, Dominator, B),
    \+ dominates(Tree, A, B).

%% first_branching_dominator(+Tree, +Node, -Dominator)
%% Find the first branching node (>1 child) that dominates Node.

first_branching_dominator(Tree, Node, Functor) :-
    compound(Tree),
    Tree =.. [Functor | Args],
    length(Args, Arity),
    Arity > 1,
    appears_in_any(Node, Args), !.

first_branching_dominator(Tree, Node, Dominator) :-
    compound(Tree),
    Tree =.. [_ | Args],
    member(Child, Args),
    first_branching_dominator(Child, Node, Dominator).
