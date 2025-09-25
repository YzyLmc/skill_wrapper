(define (problem dorfl_problem) (:domain dorfl_skillwrapper)
(:objects
	Robot - Robot
	Knife - Knife
	PeanutButterJar - PeanutButterJar
	Bread - Bread
)

(:init
	(InLeftGripper Robot PeanutButterJar)
	(InRightGripper Robot Knife)
	(LidOff PeanutButterJar)
	(HeldByRobot Robot PeanutButterJar)
)

(:goal (and
	(InLeftGripper Robot PeanutButterJar)
	(RightGripperEmpty Robot)
	(LidOff PeanutButterJar)
	(UtensilOnTable Knife)
	(Coated Knife)
	(SpreadOn Bread)
	(HeldByRobot Robot PeanutButterJar)
))

)
