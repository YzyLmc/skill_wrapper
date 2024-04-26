(define (domain kitchen_tasks)
    (:requirements :strips :typing)
    (:types location item storage)

    (:predicates
        (at ?l - location)
        (is-open ?s - storage)
        (is-closed ?s - storage)
        (holding ?i - item)
        (hand-empty)
        (in-storage ?i - item ?s - storage)
        (storage-at ?s - storage ?l - location)
        (item-at ?i - item ?l - location)
    )

    (:action GoTo
        :parameters (?from ?to - location)
        :precondition (at ?from)
        :effect (and (not (at ?from)) (at ?to))
    )

    (:action PickUp
        :parameters (?i - item ?l - location)
        :precondition (and (at ?l) (item-at ?i ?l) (not (holding ?i)) (hand-empty))
        :effect (and (not (item-at ?i ?l)) (holding ?i) (not (hand-empty)))
    )

    (:action Open
        :parameters (?s - storage ?l - location)
        :precondition (and (is-closed ?s) (storage-at ?s ?l) (at ?l))
        :effect (and (not (is-closed ?s)) (is-open ?s))
    )

    (:action PutIn
        :parameters (?i - item ?s - storage ?l - location)
        :precondition (and (holding ?i) (is-open ?s) (at ?l))
        :effect (and (not (holding ?i)) (in-storage ?i ?s) (hand-empty))
    )

    (:action Close
        :parameters (?s - storage ?l - location)
        :precondition (and (is-open ?s) (storage-at ?s ?l) (at ?l))
        :effect (is-closed ?s)
    )
)