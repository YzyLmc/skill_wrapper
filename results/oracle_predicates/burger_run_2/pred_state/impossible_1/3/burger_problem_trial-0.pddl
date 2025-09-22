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
    (iscooked Patty)
	(nothing Robot)
	(item_on Lettuce CuttingBoard)
	(item_on BottomBun Stove)
	(clear BottomBun)
	(clear Patty)
	(atop TopBun Lettuce)
	(atop Patty TopBun)
))

)
