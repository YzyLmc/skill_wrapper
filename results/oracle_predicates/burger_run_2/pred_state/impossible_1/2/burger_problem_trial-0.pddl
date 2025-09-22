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
	(item_on Lettuce CuttingBoard)
	(item_on Patty Stove)
	(clear TopBun)
	(clear BottomBun)
	(atop TopBun Lettuce)
	(atop BottomBun Patty)
)

(:goal (and
    (nothing Robot)
	(item_on Lettuce Stove)
	(item_on TopBun CuttingBoard)
	(clear BottomBun)
	(clear Patty)
	(atop BottomBun Lettuce)
	(atop Patty TopBun)
))

)
