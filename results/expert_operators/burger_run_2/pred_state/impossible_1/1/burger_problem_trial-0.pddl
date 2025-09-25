(define (problem burger_problem) (:domain burger_skillwrapper)
(:objects
    Robot - Robot
    TopBun - TopBun
    BottomBun - BottomBun
    Lettuce - Lettuce
    Patty - Patty
    Stove - Stove
    CuttingBoard - CuttingBoard
)

(:init
	(hand_empty )
	(obj_free TopBun)
	(obj_free BottomBun)
	(is_on_top TopBun Lettuce)
	(is_on_top BottomBun Patty)
	(is_on_station Lettuce CuttingBoard)
	(is_on_station Patty Stove)
)

(:goal (and
    (hand_empty )
	(obj_free TopBun)
	(obj_free BottomBun)
	(obj_free Patty)
	(is_on_top BottomBun Lettuce)
	(is_on_station Lettuce CuttingBoard)
	(is_on_station Patty Stove)
	(is_cooked Patty)
))

)
