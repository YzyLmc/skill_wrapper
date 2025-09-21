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
	(is_on_top TopBun BottomBun)
	(is_on_station Patty CuttingBoard)
)

(:goal (and
    (hand_empty )
	(obj_free TopBun)
	(obj_free Patty)
	(station_free Stove)
	(is_on_top TopBun BottomBun)
	(is_on_top Patty Lettuce)
	(is_on_station Lettuce CuttingBoard)
	(is_cut Lettuce)
))

)
