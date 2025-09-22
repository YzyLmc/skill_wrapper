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
	(nothing Robot)
	(station_empty Stove)
	(station_empty CuttingBoard)
	(clear Lettuce)
	(clear Patty)
	(atop Lettuce TopBun)
	(atop Patty BottomBun)
)

(:goal (and
    (iscooked Patty)
	(nothing Robot)
	(station_empty CuttingBoard)
	(item_on Lettuce Stove)
	(clear Lettuce)
	(clear TopBun)
	(atop TopBun Patty)
	(atop Patty BottomBun)
))

)
