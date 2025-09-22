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
	(item_on Lettuce Stove)
	(item_on Patty CuttingBoard)
	(clear Lettuce)
	(clear TopBun)
	(clear BottomBun)
	(clear Patty)
)

(:goal (and
    (iscooked Patty)
	(iscut Lettuce)
	(item_on Lettuce CuttingBoard)
	(item_on Patty Stove)
	(clear Lettuce)
	(clear TopBun)
	(clear BottomBun)
	(clear Patty)
))

)
