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
	(obj_free Lettuce)
	(obj_free TopBun)
	(obj_free BottomBun)
	(obj_free Patty)
	(station_free Stove)
	(is_on_station Lettuce CuttingBoard)
)

(:goal 
    (hand_empty )
	(is_holding TopBun)
	(is_holding Patty)
	(obj_free TopBun)
	(station_free Stove)
	(station_free CuttingBoard)
	(is_on_top Lettuce BottomBun)
	(is_on_top Lettuce Patty)
	(is_on_top TopBun Lettuce)
	(is_on_top TopBun BottomBun)
	(is_on_top TopBun Patty)
	(is_on_top BottomBun Lettuce)
	(is_on_top BottomBun Patty)
	(is_on_top Patty Lettuce)
)

)
