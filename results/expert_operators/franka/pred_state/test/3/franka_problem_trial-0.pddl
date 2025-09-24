(define (problem franka_problem) (:domain franka_skillwrapper)
(:objects
	Robot - Robot
	Plate - Plate
	Mug - Mug
	Teapot - Teapot
	Sponge - Sponge
	Bowl - Bowl
)

(:init
	(hand_empty Robot)
	(is_clean Plate)
	(is_on_top Sponge Plate)
)

(:goal (and
	(hand_empty Robot)
	(is_clean Plate)
	(is_on_top Teapot Plate)
))

)
