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
	(obj_free Patty)
	(station_free Stove)
	(station_free CuttingBoard)
	(is_on_top TopBun BottomBun)
)

(:goal (and
    (hand_empty )
	(obj_free Lettuce)
	(obj_free Patty)
	(station_free Stove)
	(station_free CuttingBoard)
	(is_on_top TopBun BottomBun)
	(is_on_top Patty TopBun)
	(is_cooked Patty)
))

)
