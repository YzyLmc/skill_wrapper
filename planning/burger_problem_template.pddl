(define (problem burger_problem) (:domain <domain>)
(:objects
    tb - TopBun
    bb - BottomBun
    l - Lettuce
    p - Patty
    s - Stove
    b - Board
)

(:init
	<init_state>
)

(:goal 
    (and
        ; the patty is cooked:
        (is_cooked p)
        ; the lettuce is sliced:
        (is_cut l)
        ; the top bun is on the lettuce:
        (is_on_top tb l)
        ; the lettuce is on top of the patty:
        (is_on_top l p)
        ; the patty is on top of the bottom bun:
        (is_on_top p bb)
    )    
)

)
