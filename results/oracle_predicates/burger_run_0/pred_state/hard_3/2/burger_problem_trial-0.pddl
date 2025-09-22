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
	(item_on TopBun CuttingBoard)
	(item_on Patty Stove)
	(clear Lettuce)
	(clear BottomBun)
	(clear Patty)
	(atop Lettuce TopBun)
)

(:goal (and
    (iscut Lettuce)
	(item_on Lettuce CuttingBoard)
	(item_on Patty Stove)
	(clear TopBun)
	(clear BottomBun)
	(atop TopBun Lettuce)
	(atop BottomBun Patty)
))

)
