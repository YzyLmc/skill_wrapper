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
	(iscooked Patty)
	(nothing Robot)
	(station_empty Stove)
	(station_empty CuttingBoard)
	(clear TopBun)
	(clear Patty)
	(atop TopBun BottomBun)
	(atop Patty Lettuce)
)

(:goal (and
    (iscooked Patty)
	(nothing Robot)
	(station_empty Stove)
	(item_on BottomBun CuttingBoard)
	(clear Lettuce)
	(clear TopBun)
	(atop TopBun Patty)
	(atop Patty BottomBun)
))

)
