(define (problem dorfl_problem) (:domain dorfl_skillwrapper)
(:objects
	Robot - Robot
	Knife - Knife
	PeanutButterJar - PeanutButterJar
	Bread - Bread
)

(:init
	(RightGripperEmpty Robot)
	(LeftGripperEmpty Robot)
	(InContainer Knife)
	(OpenableOnTable PeanutButterJar)
	(Closed PeanutButterJar)
	(Coated Knife)
)

(:goal (and
	(InLeftGripper Robot PeanutButterJar)
	(InRightGripper Robot Knife)
	(LidOff PeanutButterJar)
	(HeldByRobot Robot PeanutButterJar)
))

)
