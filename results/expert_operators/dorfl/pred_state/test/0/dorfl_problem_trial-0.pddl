(define (problem dorfl_problem) (:domain dorfl_skillwrapper)
(:objects
	Robot - Robot
	Knife - Knife
	PeanutButterJar - PeanutButterJar
	Bread - Bread
)

(:init
	(lefthand_empty Robot)
	(righthand_empty Robot)
	(is_clean Knife)
	(is_upright Knife)
)

(:goal (and
	(in_lefthand Robot PeanutButterJar)
	(in_righthand Robot Knife)
	(spread_on_top Bread)
))

)
