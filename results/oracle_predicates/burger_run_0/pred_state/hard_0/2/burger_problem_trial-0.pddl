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
	(iscut Lettuce)
	(nothing Robot)
	(station_empty Stove)
	(item_on Lettuce CuttingBoard)
	(clear TopBun)
	(clear Patty)
	(atop TopBun BottomBun)
	(atop Patty Lettuce)
))

)
