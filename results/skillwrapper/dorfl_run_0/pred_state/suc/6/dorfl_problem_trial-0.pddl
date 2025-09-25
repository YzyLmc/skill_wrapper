(define (problem dorfl_problem) (:domain dorfl_skillwrapper)
(:objects
	Robot - Robot
	Knife - Knife
	PeanutButterJar - PeanutButterJar
	Bread - Bread
)

(:init
	(InLeftGripper Robot PeanutButterJar)
	(RightGripperEmpty Robot)
	(InContainer Knife)
	(Closed PeanutButterJar)
	(HeldByRobot Robot PeanutButterJar)
)

(:goal (and
	(InLeftGripper Robot PeanutButterJar)
	(RightGripperEmpty Robot)
	(LidOff PeanutButterJar)
	(UtensilOnTable Knife)
	(Coated Knife)
	(HeldByRobot Robot PeanutButterJar)
))

)
