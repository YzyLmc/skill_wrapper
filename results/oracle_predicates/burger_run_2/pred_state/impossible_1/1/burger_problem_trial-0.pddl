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
    (iscooked Patty)
	(iscut Lettuce)
	(item_on Lettuce CuttingBoard)
	(item_on Patty Stove)
	(clear TopBun)
	(clear BottomBun)
	(clear Patty)
	(atop BottomBun Lettuce)
	(has_item Robot TopBun)
))

)
