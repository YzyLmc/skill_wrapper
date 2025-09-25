(define (problem dorfl_problem) (:domain dorfl_skillwrapper)
(:objects
	Robot - Robot
	Knife - Knife
	PeanutButterJar - PeanutButterJar
	Bread - Bread
)

(:init
	(in_lefthand Robot PeanutButterJar)
	(in_righthand Robot Knife)
	(is_opened PeanutButterJar)
	(is_clean Knife)
	(is_upright Knife)
)

(:goal (and
	(righthand_empty Robot)
	(in_lefthand Robot PeanutButterJar)
	(is_opened PeanutButterJar)
))

)
