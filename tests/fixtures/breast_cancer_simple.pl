% Breast Cancer Wisconsin — simple scenario in Prolog form.
% Mirrors the predicates and rule that tests/test_bcw_e2e.py defines manually.
%
% Feature columns after dropping "Sample code number" and selecting "Malignant" as target:
%   0  Clump Thickness
%   1  Uniformity of Cell Size
%   2  Uniformity of Cell Shape
%   3  Marginal Adhesion
%   4  Single Epithelial Cell Size
%   5  Bare Nuclei
%   6  Bland Chromatin
%   7  Normal Nucleoli
%   8  Mitoses

predicate(high_clump_thickness, 0, 3.0, greater).
predicate(low_mitoses,          8, 1.5, less).

target(X) :- high_clump_thickness(X), low_mitoses(X).
