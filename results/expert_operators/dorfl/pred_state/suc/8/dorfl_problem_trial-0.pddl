(define (problem dorfl_problem) (:domain dorfl_skillwrapper)
(:objects
	Robot - Robot
	Knife - Knife
	PeanutButterJar - PeanutButterJar
	Bread - Bread
)

(:init
	(righthand_empty Robot)
	(in_lefthand Robot PeanutButterJar)
	(is_upright Knife)
)

(:goal (and
	(in_lefthand Robot PeanutButterJar)
	(in_righthand Robot Knife)
	(is_opened PeanutButterJar)
	(is_upright Knife)
	(contains_ingredient Knife)
	(spread_on_top Bread)
))

)
