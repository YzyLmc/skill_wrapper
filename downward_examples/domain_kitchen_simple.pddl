(define (domain kitchen_tasks)
    (:requirements :strips :typing)
    (:types location item)

    (:predicates
        (at ?l - location)
        (holding ?i - item)
        (hand-empty)
        (item-at ?i - item ?l - location)
    )

    (:action GoTo
        :parameters (?to - location)
        :precondition ()
        :effect ((at ?to))
    )

    (:action PickUp
        :parameters (?i - item ?l - location)
        :precondition (and (at ?l) (item-at ?i ?l) (not (holding ?i)) (hand-empty))
        :effect (and (not (item-at ?i ?l)) (holding ?i) (not (hand-empty)))
    )

    (:action DropAt
        :parameters (?i - item ?l - location)
        :precondition (and (holding ?i) (at ?l))
        :effect (and (not (holding ?i)) (item-at ?i ?l) (hand-empty))
    )
)