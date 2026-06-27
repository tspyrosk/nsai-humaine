; NSAI Rules Export — Breast Cancer Wisconsin example, CLIPS form.
;
; Declarations are ordered facts:
;   (predicate Name ColumnIndex Threshold Comparison)
;   (composite Name (and ...))            ; and/or/not over predicate names
; Rules are defrule constructs; conditional elements are ANDed, with
; (and ...)/(or ...)/(not ...) for explicit grouping:
;   (defrule Name <CE> ... => (assert (Target ?x)))

; ---- Simple predicates ----
(predicate clump_thickness   0 0.5 greater)
(predicate uniform_cell_size 1 0.5 less)
(predicate mitoses           8 0.5 greater)

; ---- Composite predicates ----
(composite high_risk (and clump_thickness mitoses))

; ---- Rules ----
(defrule r1
    (clump_thickness ?x)
    (mitoses ?x)
    => (assert (target ?x)))

(defrule r2
    (and (or (uniform_cell_size ?x) (mitoses ?x))
         (not (clump_thickness ?x)))
    => (assert (target ?x)))
