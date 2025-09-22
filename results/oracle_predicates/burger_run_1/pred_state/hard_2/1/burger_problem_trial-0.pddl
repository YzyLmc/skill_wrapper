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
	(clear BottomBun)
	(atop Lettuce TopBun)
	(atop TopBun Patty)
)

(:goal (and
    (iscooked Patty)
	(nothing Robot)
	(station_empty Stove)
	(item_on BottomBun CuttingBoard)
	(clear Lettuce)
	(atop Lettuce TopBun)
	(atop TopBun Patty)
	(atop Patty BottomBun)
))

)
