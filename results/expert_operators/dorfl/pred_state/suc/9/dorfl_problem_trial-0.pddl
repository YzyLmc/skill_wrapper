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
	(is_upright Knife)
	(contains_ingredient Knife)
)

(:goal (and
	(righthand_empty Robot)
	(in_lefthand Robot PeanutButterJar)
	(is_opened PeanutButterJar)
	(contains_ingredient Knife)
	(spread_on_top Bread)
))

)
