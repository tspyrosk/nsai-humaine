% NSAI Rules Export — Breast Cancer Wisconsin example
%
% File format:
%   predicate(Name, ColumnIndex, Threshold, Comparison).
%     Comparison ∈ { greater, less, equal }.
%     For boolean features use Threshold = 0.5, Comparison = greater.
%
%   composite(Name, Expression).
%     Expression uses lowercase functors and/2, or/2, not/1
%     whose leaves are simple predicate names.
%
%   <target>(X) :- Body.
%     Body uses standard Prolog connectives:
%       ,    conjunction (AND)
%       ;    disjunction (OR)
%       \+   negation as failure (NOT)
%     Each atom in the body is a unary call P(X).

% ---- Simple predicates ----
predicate(clump_thickness,    0, 0.5, greater).
predicate(uniform_cell_size,  1, 0.5, less).
predicate(mitoses,            8, 0.5, greater).

% ---- Composite predicates ----
composite(high_risk, and(clump_thickness, mitoses)).

% ---- Rules ----
target(X) :- clump_thickness(X), mitoses(X).
target(X) :- (uniform_cell_size(X) ; mitoses(X)), \+ clump_thickness(X).
