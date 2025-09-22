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
	(clear BottomBun)
	(clear Patty)
	(atop BottomBun Lettuce)
	(atop Patty TopBun)
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
