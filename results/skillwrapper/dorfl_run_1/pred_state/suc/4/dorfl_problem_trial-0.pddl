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
)

(:goal (and
	(InLeftGripper Robot PeanutButterJar)
	(RightGripperEmpty Robot)
	(Closed PeanutButterJar)
	(HeldByRobot Robot PeanutButterJar)
))

)
