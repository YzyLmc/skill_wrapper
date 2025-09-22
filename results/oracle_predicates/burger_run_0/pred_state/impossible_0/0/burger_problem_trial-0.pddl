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
	(item_on Lettuce CuttingBoard)
	(item_on Patty Stove)
	(clear Lettuce)
	(clear TopBun)
	(clear BottomBun)
	(clear Patty)
)

(:goal (and
    (iscooked Patty)
	(nothing Robot)
	(station_empty Stove)
	(item_on Lettuce CuttingBoard)
	(clear Lettuce)
	(clear TopBun)
	(clear Patty)
	(atop Patty BottomBun)
))

)
