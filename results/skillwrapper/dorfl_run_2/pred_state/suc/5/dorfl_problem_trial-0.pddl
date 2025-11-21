(define (problem dorfl_problem) (:domain dorfl_skillwrapper)
(:objects
	Robot - Robot
	Knife - Knife
	PeanutButterJar - PeanutButterJar
	Bread - Bread
)

(:init
	(InRightGripper Robot Knife)
	(LeftGripperEmpty Robot)
	(OpenableOnTable PeanutButterJar)
	(Closed PeanutButterJar)
)

(:goal (and
	(InLeftGripper Robot PeanutButterJar)
	(RightGripperEmpty Robot)
	(LidOff PeanutButterJar)
	(UtensilOnTable Knife)
))

)
