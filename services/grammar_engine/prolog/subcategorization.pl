:- module(subcategorization, [subcat_frame/2, check_theta_criterion/3, role_assignment/3]).

:- use_module(lexicon).

%% ==========================================================================
%% MLAF Grammar Engine — Theta Grids & Argument Structure
%%
%% Implements the Theta Criterion (Chomsky 1981):
%%   - Each argument receives exactly one theta-role
%%   - Each theta-role is assigned to exactly one argument
%% ==========================================================================

%% --- subcat_frame(+VerbID, -Frame) ---
%% Returns the subcategorization frame for a verb.
%% Frame = required(Roles) where Roles is the theta grid.

subcat_frame(VerbID, required(Roles)) :-
    theta_grid(VerbID, Roles).

subcat_frame(VerbID, none) :-
    \+ theta_grid(VerbID, _).

%% --- check_theta_criterion(+VerbID, +Arguments, -Result) ---
%% Arguments is a list of argument IDs (gesture IDs for subject and object).
%% Returns satisfied or violation(Type, Count).

check_theta_criterion(VerbID, Arguments, satisfied) :-
    theta_grid(VerbID, Roles),
    length(Roles, NRoles),
    length(Arguments, NArgs),
    NRoles =:= NArgs,
    !.

check_theta_criterion(VerbID, Arguments, violation(missing_args, Missing)) :-
    theta_grid(VerbID, Roles),
    length(Roles, NRoles),
    length(Arguments, NArgs),
    NArgs < NRoles,
    Missing is NRoles - NArgs,
    !.

check_theta_criterion(VerbID, Arguments, violation(extra_args, Extra)) :-
    theta_grid(VerbID, Roles),
    length(Roles, NRoles),
    length(Arguments, NArgs),
    NArgs > NRoles,
    Extra is NArgs - NRoles,
    !.

check_theta_criterion(VerbID, _, violation(no_frame, 0)) :-
    \+ theta_grid(VerbID, _).

%% --- role_assignment(+VerbID, +Position, -Role) ---
%% Maps verb + argument position to thematic role.
%% Position: 1 = external argument (subject), 2 = internal argument (object)

role_assignment(VerbID, Position, Role) :-
    theta_grid(VerbID, Roles),
    nth1(Position, Roles, Role).

role_assignment(_, Position, unassigned) :-
    Position > 2.
