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
	(clear TopBun)
	(atop Lettuce Patty)
	(atop TopBun BottomBun)
)

(:goal (and
    (iscooked Patty)
	(iscut Lettuce)
	(nothing Robot)
	(station_empty Stove)
	(station_empty CuttingBoard)
	(clear TopBun)
	(atop Lettuce BottomBun)
	(atop TopBun Patty)
	(atop Patty Lettuce)
))

)
