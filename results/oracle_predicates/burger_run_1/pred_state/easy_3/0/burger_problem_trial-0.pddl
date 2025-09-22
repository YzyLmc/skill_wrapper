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
	(item_on Patty CuttingBoard)
	(clear Lettuce)
	(clear TopBun)
	(clear BottomBun)
	(clear Patty)
)

(:goal (and
    (iscooked Patty)
	(iscut Lettuce)
	(nothing Robot)
	(station_empty Stove)
	(item_on Lettuce CuttingBoard)
	(item_on TopBun CuttingBoard)
	(clear Lettuce)
	(clear TopBun)
	(clear Patty)
	(atop Patty BottomBun)
))

)
