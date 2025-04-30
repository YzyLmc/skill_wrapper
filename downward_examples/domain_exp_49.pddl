(define (domain exp_49)
    (:requirements :strips :typing)
    (:types location object)
    (:predicates
        (hasemptyhand)
        (isat ?o - object ?l - location)
        (isclose ?l - location)
        (isgrasped ?o - object)
        (isloose ?o - object)
        (isunoccupied ?l - location)
    )
(:action DropAt_1
    :parameters (?l - location ?l2 - location ?l1 - location ?o - object)
    :precondition (and (not (hasemptyhand )) (not (isat ?o ?l)) (isclose ?l) (isgrasped ?o) (isloose ?o) (not (isunoccupied ?l)))
    :effect (and (hasemptyhand ) (isat ?o ?l) (not (isgrasped ?o)))
)

(:action DropAt_2
    :parameters (?l - location ?l2 - location ?l1 - location ?o - object)
    :precondition (and (hasemptyhand ) (not (isat ?o ?l)) (isclose ?l) (isgrasped ?o) (isloose ?o) (not (isunoccupied ?l)))
    :effect (and (isat ?o ?l) (not (isgrasped ?o)))
)

(:action GoTo_1
    :parameters (?o - object ?l - location ?l2 - location ?l1 - location)
    :precondition (and (hasemptyhand ) (isclose ?l1) (not (isclose ?l2)))
    :effect (and (not (isclose ?l1)) (isclose ?l2))
)

(:action GoTo_2
    :parameters (?o - object ?l - location ?l2 - location ?l1 - location)
    :precondition (and (not (hasemptyhand )) (isclose ?l1) (not (isclose ?l2)))
    :effect (and (hasemptyhand ) (not (isclose ?l1)) (isclose ?l2))
)

(:action GoTo_3
    :parameters (?o - object ?l - location ?l2 - location ?l1 - location)
    :precondition (and (hasemptyhand ) (isclose ?l1) (not (isclose ?l2)))
    :effect (and (not (hasemptyhand )) (not (isclose ?l1)) (isclose ?l2))
)

(:action PickUp_1
    :parameters (?l - location ?l2 - location ?l1 - location ?o - object)
    :precondition (and (hasemptyhand ) (isat ?o ?l) (isclose ?l) (not (isgrasped ?o)) (isloose ?o) (not (isunoccupied ?l)))
    :effect (and (not (hasemptyhand )) (not (isat ?o ?l)) (isgrasped ?o))
)

(:action PickUp_2
    :parameters (?l - location ?l2 - location ?l1 - location ?o - object)
    :precondition (and (hasemptyhand ) (not (isat ?o ?l)) (isclose ?l) (not (isgrasped ?o)) (not (isloose ?o)) (not (isunoccupied ?l)))
    :effect (and (not (hasemptyhand )) (isgrasped ?o) (isloose ?o))
)

(:action PickUp_3
    :parameters (?l - location ?l2 - location ?l1 - location ?o - object)
    :precondition (and (hasemptyhand ) (isat ?o ?l) (isclose ?l) (not (isgrasped ?o)) (isloose ?o) (not (isunoccupied ?l)))
    :effect (and (not (isat ?o ?l)) (not (isloose ?o)) (isunoccupied ?l))
)

(:action PickUp_4
    :parameters (?l - location ?l2 - location ?l1 - location ?o - object)
    :precondition (and (hasemptyhand ) (isat ?o ?l) (isclose ?l) (not (isgrasped ?o)) (isloose ?o) (not (isunoccupied ?l)))
    :effect (and (not (hasemptyhand )) (not (isat ?o ?l)) (isgrasped ?o) (isunoccupied ?l))
)

(:action PickUp_5
    :parameters (?l - location ?l2 - location ?l1 - location ?o - object)
    :precondition (and (not (hasemptyhand )) (isat ?o ?l) (isclose ?l) (not (isgrasped ?o)) (isloose ?o) (not (isunoccupied ?l)))
    :effect (and (not (isat ?o ?l)) (isgrasped ?o) (isunoccupied ?l))
)

)