(define (problem burger_problem) (:domain <domain>)
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
	(station_free Stove)
	(station_free CuttingBoard)
	(is_on_top Lettuce TopBun)
	(is_on_top TopBun Lettuce)
	(is_on_top BottomBun Patty)
	(is_on_top Patty BottomBun)
)

(:goal 
    (hand_empty )
	(obj_free Lettuce)
	(obj_free BottomBun)
	(station_free Stove)
	(station_free CuttingBoard)
	(is_on_top Lettuce TopBun)
	(is_on_top Lettuce BottomBun)
	(is_on_top TopBun Patty)
	(is_on_top BottomBun Patty)
	(is_on_top Patty BottomBun)
	(is_cooked Patty)
)

)
