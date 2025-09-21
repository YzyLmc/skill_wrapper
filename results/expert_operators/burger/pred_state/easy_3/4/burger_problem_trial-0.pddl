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
	(obj_free Lettuce)
	(obj_free TopBun)
	(obj_free BottomBun)
	(obj_free Patty)
	(is_on_station Lettuce Stove)
	(is_on_station Patty CuttingBoard)
	(is_cooked Patty)
)

(:goal (and
    (hand_empty )
	(obj_free Lettuce)
	(obj_free TopBun)
	(obj_free BottomBun)
	(obj_free Patty)
	(is_on_station Lettuce CuttingBoard)
	(is_on_station Patty Stove)
	(is_cut Lettuce)
	(is_cooked Patty)
))

)
