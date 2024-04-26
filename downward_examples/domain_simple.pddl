(define (domain moving)
    (:requirements :strips)
    (:types room)
    (:predicates (room ?r) (robot-at ?r))

    (:action move
        :parameters (?from ?to)
        :precondition (and (room ?from) (room ?to) (robot-at ?from))
        :effect (and (not (robot-at ?from)) (robot-at ?to))
    )
)