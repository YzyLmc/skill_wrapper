(define (domain dorfl_skillwrapper)

	(:requirements :adl :typing :equality :conditional-effects)

	(:types
		Knife - utensil
		Bread - food
		Robot - robot
		robot - object
		openable - object
		PeanutButterJar - openable
		food - object
		utensil - object
	)

	(:predicates
		(InLeftGripper ?robot - robot ?openable - openable)
		(InRightGripper ?robot - robot ?utensil - utensil)
		(RightGripperEmpty ?robot - robot)
		(LidOff ?openable - openable)
		(LeftGripperEmpty ?robot - robot)
		(InContainer ?utensil - utensil)
		(OpenableOnTable ?openable - openable)
		(UtensilOnTable ?utensil - utensil)
		(Closed ?openable - openable)
		(Coated ?utensil - utensil)
		(SpreadOn ?food - food)
	)

	(:action LeftArmPick_1 
:parameters ( ?openable_p0 - openable  ?robot_p1 - robot )
:precondition (and 
	(Closed ?openable_p0 )
	(LeftGripperEmpty ?robot_p1 )
	(OpenableOnTable ?openable_p0 )
	(not (InLeftGripper ?robot_p1 ?openable_p0))
	(not (LidOff ?openable_p0 ))
) 
:effect (and 
 	(InLeftGripper ?robot_p1 ?openable_p0) 
	(not (LeftGripperEmpty ?robot_p1 ))
	(not (OpenableOnTable ?openable_p0 ))
 ) 
)


(:action RightArmPick_2 
:parameters ( ?robot_p1 - robot  ?utensil_p0 - utensil )
:precondition (and 
	(InContainer ?utensil_p0 )
	(RightGripperEmpty ?robot_p1 )
	(not (Coated ?utensil_p0 ))
	(not (InRightGripper ?robot_p1 ?utensil_p0))
	(not (UtensilOnTable ?utensil_p0 ))
) 
:effect (and 
 	(InRightGripper ?robot_p1 ?utensil_p0) 
	(not (InContainer ?utensil_p0 ))
	(not (RightGripperEmpty ?robot_p1 ))
 ) 
)


(:action Drop_3 
:parameters ( ?robot_p1 - robot  ?utensil_p0 - utensil )
:precondition (and 
	(InRightGripper ?robot_p1 ?utensil_p0)
	(not (Coated ?utensil_p0 ))
	(not (InContainer ?utensil_p0 ))
	(not (RightGripperEmpty ?robot_p1 ))
	(not (UtensilOnTable ?utensil_p0 ))
) 
:effect (and 
 	(RightGripperEmpty ?robot_p1 ) 
	(UtensilOnTable ?utensil_p0 ) 
	(not (InRightGripper ?robot_p1 ?utensil_p0))
 ) 
)


(:action Open_4 
:parameters ( ?openable_p0 - openable  ?robot_p1 - robot )
:precondition (and 
	(Closed ?openable_p0 )
	(InLeftGripper ?robot_p1 ?openable_p0)
	(not (LeftGripperEmpty ?robot_p1 ))
	(not (LidOff ?openable_p0 ))
	(not (OpenableOnTable ?openable_p0 ))
) 
:effect (and 
 	(LidOff ?openable_p0 ) 
	(not (Closed ?openable_p0 ))
 ) 
)


(:action Scoop_5 
:parameters ( ?openable_p0 - openable  ?robot_p1 - robot  ?utensil_p0 - utensil )
:precondition (and 
	(InLeftGripper ?robot_p1 ?openable_p0)
	(InRightGripper ?robot_p1 ?utensil_p0)
	(LidOff ?openable_p0 )
	(not (Closed ?openable_p0 ))
	(not (Coated ?utensil_p0 ))
	(not (InContainer ?utensil_p0 ))
	(not (LeftGripperEmpty ?robot_p1 ))
	(not (OpenableOnTable ?openable_p0 ))
	(not (RightGripperEmpty ?robot_p1 ))
	(not (UtensilOnTable ?utensil_p0 ))
) 
:effect (and 
 	(Coated ?utensil_p0 ) 
 ) 
)


(:action Spread_6 
:parameters ( ?food_p0 - food  ?openable_p0 - openable  ?robot_p1 - robot  ?utensil_p0 - utensil )
:precondition (and 
	(Coated ?utensil_p0 )
	(InLeftGripper ?robot_p1 ?openable_p0)
	(InRightGripper ?robot_p1 ?utensil_p0)
	(not (InContainer ?utensil_p0 ))
	(not (LeftGripperEmpty ?robot_p1 ))
	(not (RightGripperEmpty ?robot_p1 ))
	(not (SpreadOn ?food_p0 ))
	(not (UtensilOnTable ?utensil_p0 ))
) 
:effect (and 
 	(SpreadOn ?food_p0 ) 
 ) 
)

)