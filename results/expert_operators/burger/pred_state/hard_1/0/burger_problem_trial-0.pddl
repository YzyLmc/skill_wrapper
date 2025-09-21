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
	(station_free Stove)
	(station_free CuttingBoard)
	(is_on_top Lettuce Patty)
)

(:goal (and
    (hand_empty )
	(obj_free TopBun)
	(station_free Stove)
	(station_free CuttingBoard)
	(is_on_top Lettuce Patty)
	(is_on_top TopBun Lettuce)
	(is_on_top Patty BottomBun)
	(is_cut Lettuce)
	(is_cooked Patty)
))

)
