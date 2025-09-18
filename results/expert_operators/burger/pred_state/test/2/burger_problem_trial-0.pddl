(define (problem burger_problem) (:domain burger_skillwrapper)
(:objects
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
	(obj_free Patty)
	(station_free Stove)
	(station_free CuttingBoard)
	(is_on_top TopBun Lettuce)
	(is_on_top Patty BottomBun)
)

(:goal (and
    (hand_empty )
	(obj_free Lettuce)
	(obj_free Patty)
	(station_free Stove)
	(station_free CuttingBoard)
	(is_on_top Lettuce TopBun)
	(is_on_top Patty BottomBun)
	(is_cut Lettuce)
	(is_cooked Patty)
))

)
