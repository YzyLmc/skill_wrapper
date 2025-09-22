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
	(obj_free BottomBun)
	(obj_free Patty)
	(station_free Stove)
	(station_free CuttingBoard)
	(is_on_top BottomBun Lettuce)
	(is_on_top Patty TopBun)
	(is_cut Lettuce)
)

(:goal (and
    (hand_empty )
	(obj_free TopBun)
	(obj_free Patty)
	(is_on_top Lettuce BottomBun)
	(is_on_top TopBun Lettuce)
	(is_on_station BottomBun Stove)
	(is_on_station Patty CuttingBoard)
	(is_cut Lettuce)
))

)
