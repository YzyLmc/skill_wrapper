(define (domain exp_33)
    (:requirements :strips :typing)
    (:types item location)
    (:predicates
        (at ?l - location)
        (canfit ?i - item)
        (holding ?i - item)
        (isfixed ?i - item)
        (isstableplacement ?i - item)
        (near ?i - item)
        (near ?l - location)
    )
(:action DropAt_1
    :parameters (?o - object ?l1 - location ?l2 - location ?l - location)
    :precondition (and (not (at ?l)) (not (canfit ?o ?l)) (holding ?o) (not (isfixed ?o)) (isstableplacement ?o ?l) (near ?l) (near ?o))
    :effect (and (canfit ?o ?l) (not (holding ?o)) (not (isstableplacement ?o ?l)) (not (near ?o)))
)

(:action DropAt_2
    :parameters (?o - object ?l2 - location ?l - location ?l1 - location)
    :precondition (and (canfit ?o ?l) (holding ?o) (not (isfixed ?o)) (isstableplacement ?o ?l) (near ?l) (near ?o))
    :effect (and (not (holding ?o)) (isfixed ?o))
)

(:action DropAt_3
    :parameters (?o - object ?l1 - location ?l2 - location ?l - location)
    :precondition (and (at ?l) (canfit ?o ?l) (holding ?o) (not (isfixed ?o)) (isstableplacement ?o ?l) (near ?l) (near ?o))
    :effect (and (not (at ?l)) (not (holding ?o)) (isfixed ?o))
)

(:action GoTo_1
    :parameters (?o - object ?l1 - location ?l2 - location ?l - location)
    :precondition (and (at ?l1) (not (at ?l2)) (near ?l1) (not (near ?l2)))
    :effect (and (not (at ?l1)) (not (near ?l1)) (near ?l2))
)

(:action GoTo_2
    :parameters (?o - object ?l1 - location ?l2 - location ?l - location)
    :precondition (and (at ?l1) (not (at ?l2)) (near ?l1) (not (near ?l2)))
    :effect (and (not (at ?l1)) (near ?l2))
)

(:action GoTo_3
    :parameters (?o - object ?l1 - location ?l2 - location ?l - location)
    :precondition (and (at ?l1) (not (at ?l2)) (near ?l1) (not (near ?l2)))
    :effect (and (not (at ?l1)) (at ?l2) (not (near ?l1)) (near ?l2))
)

(:action PickUp_1
    :parameters (?o - object ?l1 - location ?l2 - location ?l - location)
    :precondition (and (at ?l) (canfit ?o ?l) (not (holding ?o)) (isfixed ?o) (isstableplacement ?o ?l) (near ?l) (near ?o))
    :effect (and (not (at ?l)) (holding ?o) (not (isfixed ?o)) (not (isstableplacement ?o ?l)))
)

(:action PickUp_2
    :parameters (?o - object ?l1 - location ?l2 - location ?l - location)
    :precondition (and (not (at ?l)) (canfit ?o ?l) (not (holding ?o)) (isfixed ?o) (isstableplacement ?o ?l) (near ?l) (near ?o))
    :effect (and (at ?l) (holding ?o) (not (isfixed ?o)))
)

(:action PickUp_3
    :parameters (?o - object ?l1 - location ?l2 - location ?l - location)
    :precondition (and (at ?l) (canfit ?o ?l) (not (holding ?o)) (isfixed ?o) (isstableplacement ?o ?l) (near ?l) (near ?o))
    :effect (and (holding ?o) (not (isfixed ?o)))
)

(:action PickUp_4
    :parameters (?o - object ?l1 - location ?l2 - location ?l - location)
    :precondition (and (at ?l) (canfit ?o ?l) (not (holding ?o)) (isfixed ?o) (isstableplacement ?o ?l) (near ?l) (near ?o))
    :effect (and (not (canfit ?o ?l)) (not (isstableplacement ?o ?l)) (not (near ?o)))
)

(:action PickUp_5
    :parameters (?o - object ?l1 - location ?l2 - location ?l - location)
    :precondition (and (not (at ?l)) (canfit ?o ?l) (not (holding ?o)) (isfixed ?o) (isstableplacement ?o ?l) (near ?l) (near ?o))
    :effect (and (holding ?o) (not (isfixed ?o)) (not (isstableplacement ?o ?l)))
)

(:action PickUp_6
    :parameters (?o - object ?l1 - location ?l2 - location ?l - location)
    :precondition (and (not (at ?l)) (canfit ?o ?l) (not (holding ?o)) (isfixed ?o) (isstableplacement ?o ?l) (near ?l) (near ?o))
    :effect (and (not (canfit ?o ?l)) (holding ?o) (not (isfixed ?o)) (not (isstableplacement ?o ?l)))
)

)