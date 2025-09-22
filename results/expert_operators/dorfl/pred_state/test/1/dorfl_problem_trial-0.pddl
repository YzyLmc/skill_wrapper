(define (problem dorfl_problem) (:domain dorfl_skillwrapper)
(:objects
	Robot - Robot
	Knife - Knife
	PeanutButterJar - PeanutButterJar
	Bread - Bread
)

(:init
	(lefthand_empty Robot)
	(in_righthand Robot Knife)
	(is_clean Knife)
)

(:goal (and
	(in_lefthand Robot PeanutButterJar)
	(in_righthand Robot Knife)
	(spread_on_top Bread)
))

)
