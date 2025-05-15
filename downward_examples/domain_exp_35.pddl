(define (domain exp_35)
    (:requirements :strips :typing)
    (:types object location)
    (:predicates
        (canhold ?o - object ?l - location)
        (cansupport ?o - object ?l - location)
        (isaccessible ?l - location)
        (isat ?o - object ?l - location)
        (isempty ?l - location)
        (isholding ?o - object)
        (ismovable ?o - object)
        (isnear ?l - location)
        (isopen ?l - location)
    )
(:action DropAt_1
    :parameters (?l1 - location ?l - location ?o - object ?l2 - location)
    :precondition (and (not (canhold ?o ?l)) (cansupport ?l ?o) (isaccessible ?l) (not (isat ?o ?l)) (not (isempty ?l)) (not (isholding ?o)) (not (ismovable ?o)) (isnear ?l) (not (isopen ?l)))
    :effect (and (canhold ?o ?l) (isat ?o ?l) (isopen ?l))
)

(:action DropAt_2
    :parameters (?l1 - location ?l - location ?o - object ?l2 - location)
    :precondition (and (canhold ?o ?l) (cansupport ?l ?o) (isaccessible ?l) (not (isat ?o ?l)) (isempty ?l) (isholding ?o) (ismovable ?o) (isnear ?l) (not (isopen ?l)))
    :effect (and (isat ?o ?l) (not (isempty ?l)) (not (isholding ?o)) (not (ismovable ?o)))
)

(:action DropAt_3
    :parameters (?l1 - location ?l - location ?o - object ?l2 - location)
    :precondition (and (canhold ?o ?l) (cansupport ?l ?o) (isaccessible ?l) (not (isat ?o ?l)) (not (isempty ?l)) (isholding ?o) (ismovable ?o) (isnear ?l) (not (isopen ?l)))
    :effect (and (isat ?o ?l) (not (isholding ?o)) (not (ismovable ?o)) (isopen ?l))
)

(:action DropAt_4
    :parameters (?l1 - location ?l - location ?o - object ?l2 - location)
    :precondition (and (canhold ?o ?l) (cansupport ?l ?o) (isaccessible ?l) (not (isat ?o ?l)) (not (isempty ?l)) (isholding ?o) (ismovable ?o) (isnear ?l) (isopen ?l))
    :effect (and (isat ?o ?l) (not (isholding ?o)) (not (ismovable ?o)))
)

(:action DropAt_5
    :parameters (?l1 - location ?l - location ?o - object ?l2 - location)
    :precondition (and (canhold ?o ?l) (cansupport ?l ?o) (isaccessible ?l) (not (isat ?o ?l)) (isempty ?l) (not (isholding ?o)) (not (ismovable ?o)) (isnear ?l) (not (isopen ?l)))
    :effect (and (isat ?o ?l) (not (isempty ?l)))
)

(:action GoTo_1
    :parameters (?l1 - location ?l - location ?o - object ?l2 - location)
    :precondition (and (not (isaccessible ?l2)) (isnear ?l1) (not (isnear ?l2)))
    :effect (and (isaccessible ?l2) (isnear ?l2))
)

(:action GoTo_2
    :parameters (?l1 - location ?l - location ?o - object ?l2 - location)
    :precondition (and (not (isaccessible ?l2)) (isnear ?l1) (isnear ?l2))
    :effect (and (isaccessible ?l2))
)

(:action GoTo_3
    :parameters (?l1 - location ?l - location ?o - object ?l2 - location)
    :precondition (and (not (isaccessible ?l2)) (isnear ?l1) (isnear ?l2))
    :effect (and (isaccessible ?l2) (not (isnear ?l1)))
)

(:action GoTo_4
    :parameters (?l1 - location ?l - location ?o - object ?l2 - location)
    :precondition (and (not (isaccessible ?l2)) (isnear ?l1) (not (isnear ?l2)))
    :effect (and (isaccessible ?l2) (not (isnear ?l1)) (isnear ?l2))
)

(:action GoTo_5
    :parameters (?l1 - location ?l - location ?o - object ?l2 - location)
    :precondition (and (isaccessible ?l2) (isnear ?l1) (not (isnear ?l2)))
    :effect (and (not (isnear ?l1)) (isnear ?l2))
)

(:action PickUp_1
    :parameters (?l1 - location ?l - location ?o - object ?l2 - location)
    :precondition (and (canhold ?o ?l) (cansupport ?l ?o) (isaccessible ?l) (isat ?o ?l) (not (isempty ?l)) (not (isholding ?o)) (not (ismovable ?o)) (isnear ?l) (not (isopen ?l)))
    :effect (and (not (isat ?o ?l)) (isholding ?o) (ismovable ?o))
)

(:action PickUp_2
    :parameters (?l1 - location ?l - location ?o - object ?l2 - location)
    :precondition (and (canhold ?o ?l) (cansupport ?l ?o) (isaccessible ?l) (isat ?o ?l) (not (isempty ?l)) (not (isholding ?o)) (not (ismovable ?o)) (isnear ?l) (not (isopen ?l)))
    :effect (and (not (canhold ?o ?l)) (not (isat ?o ?l)))
)

(:action PickUp_3
    :parameters (?l1 - location ?l - location ?o - object ?l2 - location)
    :precondition (and (canhold ?o ?l) (cansupport ?l ?o) (isaccessible ?l) (isat ?o ?l) (not (isempty ?l)) (not (isholding ?o)) (ismovable ?o) (isnear ?l) (not (isopen ?l)))
    :effect (and (not (isat ?o ?l)) (isempty ?l) (isholding ?o) (isopen ?l))
)

(:action PickUp_4
    :parameters (?l1 - location ?l - location ?o - object ?l2 - location)
    :precondition (and (canhold ?o ?l) (cansupport ?l ?o) (isaccessible ?l) (isat ?o ?l) (not (isempty ?l)) (not (isholding ?o)) (not (ismovable ?o)) (isnear ?l) (not (isopen ?l)))
    :effect (and (not (isat ?o ?l)) (isempty ?l) (isholding ?o) (ismovable ?o))
)

)