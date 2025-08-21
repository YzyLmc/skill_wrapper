(define (problem dorfl_problem_1) (:domain skillwrapper)
(:objects
	k - Knife
	j - Jar
	pb - PeanutButter
	b - Bread
	left_gripper - arm
	right_gripper - arm
	t - Table
)

(:init
	<init_state>
)

(:goal (and
	; peanut butter is spread on the bread:
	(is_spread pb b)
))

)
