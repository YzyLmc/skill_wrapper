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
	(clear Lettuce)
	(clear TopBun)
	(atop TopBun BottomBun)
	(atop BottomBun Patty)
)

(:goal (and
    (nothing Robot)
	(item_on Lettuce CuttingBoard)
	(item_on TopBun CuttingBoard)
	(item_on BottomBun Stove)
	(clear Lettuce)
	(clear BottomBun)
	(clear Patty)
	(atop Patty TopBun)
))

)
