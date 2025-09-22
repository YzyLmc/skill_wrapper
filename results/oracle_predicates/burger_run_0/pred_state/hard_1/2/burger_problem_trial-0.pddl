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
	(iscut Lettuce)
	(nothing Robot)
	(station_empty Stove)
	(station_empty CuttingBoard)
	(clear TopBun)
	(clear Patty)
	(atop TopBun Lettuce)
	(atop Patty BottomBun)
)

(:goal (and
    (iscut Lettuce)
	(nothing Robot)
	(item_on BottomBun Stove)
	(item_on Patty CuttingBoard)
	(clear TopBun)
	(clear Patty)
	(atop Lettuce BottomBun)
	(atop TopBun Lettuce)
))

)
