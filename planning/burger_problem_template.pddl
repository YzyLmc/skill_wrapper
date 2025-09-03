(define (problem burger_problem) (:domain <domain>)
(:objects
    TopBun - TopBun
    BottomBun - BottomBun
    Lettuce - Lettuce
    Patty - Patty
    Stove - Stove
    Board - Board
)

(:init
	<init_state>
)

(:goal 
    (and
        ; the patty is cooked:
        (is_cooked Patty)
        ; the lettuce is sliced:
        (is_cut Lettuce)
        ; the top bun is on the lettuce:
        (is_on_top TopBun Lettuce)
        ; the lettuce is on top of the patty:
        (is_on_top Lettuce Patty)
        ; the patty is on top of the bottom bun:
        (is_on_top Patty BottomBun)
    )    
)

)
