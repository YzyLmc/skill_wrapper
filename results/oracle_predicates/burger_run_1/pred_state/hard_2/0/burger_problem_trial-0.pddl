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
	(iscut Lettuce)
	(nothing Robot)
	(station_empty Stove)
	(station_empty CuttingBoard)
	(clear TopBun)
	(clear BottomBun)
	(atop BottomBun Patty)
	(atop Patty Lettuce)
)

(:goal (and
    (iscooked Patty)
	(iscut Lettuce)
	(nothing Robot)
	(station_empty Stove)
	(item_on BottomBun CuttingBoard)
	(clear TopBun)
	(atop Lettuce Patty)
	(atop TopBun Lettuce)
	(atop Patty BottomBun)
))

)
