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
	(item_on Lettuce CuttingBoard)
	(item_on Patty Stove)
	(clear TopBun)
	(clear BottomBun)
	(atop TopBun Lettuce)
	(atop BottomBun Patty)
)

(:goal (and
    (nothing Robot)
	(item_on TopBun CuttingBoard)
	(item_on Patty Stove)
	(clear Lettuce)
	(clear BottomBun)
	(atop Lettuce TopBun)
	(atop BottomBun Patty)
))

)
