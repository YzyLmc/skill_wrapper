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
	(gripper_empty Robot)
	(top_most Lettuce)
	(top_most TopBun)
	(top_most BottomBun)
	(top_most Patty)
	(station_unoccupied Stove)
	(station_unoccupied CuttingBoard)
	(cooked Patty)
)

(:goal (and
    (gripper_empty Robot)
	(on_cutting_board Lettuce)
	(cut_into_pieces Lettuce)
	(top_most Lettuce)
	(top_most TopBun)
	(top_most BottomBun)
	(top_most Patty)
	(on_station Lettuce CuttingBoard)
	(on_station Patty Stove)
	(cooked Patty)
))

)
