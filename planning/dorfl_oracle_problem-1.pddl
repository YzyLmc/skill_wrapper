(define (problem dorfl_problem_1) (:domain skillwrapper)
(:objects
	k - Knife
	j - Jar
	pb - PeanutButter
	b - Bread
	left_gripper - Gripper
	right_gripper - Gripper
	t - Table
)

(:init
	; both of the robot's grippers are empty:
	(hand_empty left_gripper)
	(hand_empty right_gripper)

	; the knife is close to the left gripper and the jar is close to the right gripper:
	(is_graspable k left_gripper)
	(is_graspable j right_gripper)

	; the jar contains peanut butter::
	(contains j pb)

	; the bread, jar, and knife are on the table:
	(on_location b t)
	(on_location k t)
	(on_location j t)
)

(:goal (and
	; peanut butter is spread on the bread:
	(is_spread pb b)
))

)
