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
	(item_on Lettuce CuttingBoard)
	(clear Lettuce)
	(clear TopBun)
	(clear BottomBun)
	(clear Patty)
)

(:goal (and
    (nothing Robot)
	(station_empty CuttingBoard)
	(item_on Lettuce Stove)
	(clear Lettuce)
	(clear TopBun)
	(clear BottomBun)
	(clear Patty)
))

)
