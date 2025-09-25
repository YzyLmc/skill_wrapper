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
	(is_clean Knife)
	(is_upright Knife)
)

(:goal (and
	(righthand_empty Robot)
	(in_lefthand Robot PeanutButterJar)
	(is_opened PeanutButterJar)
	(contains_ingredient Knife)
	(spread_on_top Bread)
))

)
