(define (domain dorfl_skillwrapper)

	(:requirements :adl :typing :equality :conditional-effects)

	(:types
		robot - object
		PeanutButterJar - openable
		food - object
		Robot - robot
		utensil - object
		Bread - food
		openable - object
		Knife - utensil
	)

	(:predicates
		(lefthand_empty ?robot - robot)
		(righthand_empty ?robot - robot)
		(in_lefthand ?robot - robot ?openable - openable)
		(in_righthand ?robot - robot ?utensil - utensil)
		(is_opened ?openable - openable)
		(is_clean ?utensil - utensil)
		(is_upright ?utensil - utensil)
		(contains_ingredient ?utensil - utensil)
		(spread_on_top ?food - food)
	)

	(:action LeftArmPick
 		:parameters (?robot - robot ?openable - openable) 
		:precondition (and   
			(lefthand_empty ?robot) 
			(not (in_lefthand ?robot ?openable)) 
		) 
		:effect (and 
			(not (lefthand_empty ?robot)) 
			(in_lefthand ?robot ?openable) 
		) 
	)

(:action RightArmPick
 		:parameters (?robot - robot ?utensil - utensil) 
		:precondition (and   
			(righthand_empty ?robot) 
			(is_upright ?utensil) 
			(not (in_righthand ?robot ?utensil)) 
		) 
		:effect (and 
			(not (righthand_empty ?robot)) 
			(not (is_upright ?utensil)) 
			(in_righthand ?robot ?utensil) 
		) 
	)

(:action Drop
 		:parameters (?robot - robot ?utensil - utensil) 
		:precondition (and   
			(not (righthand_empty ?robot)) 
			(in_righthand ?robot ?utensil) 
		) 
		:effect (and 
			(righthand_empty ?robot) 
			(not (in_righthand ?robot ?utensil)) 
		) 
	)

(:action Open
 	:parameters (?robot - robot ?openable - openable) 
		:precondition (and 
			(not (is_opened ?openable))   
			(in_lefthand ?robot ?openable) 
			(not (lefthand_empty ?robot)) 
			(righthand_empty ?robot) 
		) 
		:effect (and 
			(is_opened ?openable) 
		) 
	)

(:action Scoop
 	:parameters (?robot - robot ?knife - utensil ?jar - openable) 
		:precondition (and 
			(in_lefthand ?robot ?jar) 
			(in_righthand ?robot ?knife) 
			(is_opened ?jar) 
			(is_clean ?knife) 
			(not (contains_ingredient ?knife)) 
		) 
		:effect (and 
			(contains_ingredient ?knife) 
			(not (is_clean ?knife)) 
		) 
	)

(:action Spread
 		:parameters (?robot - robot ?knife - utensil ?bread - food) 
		:precondition (and 
			(in_righthand ?robot ?knife) 
			(contains_ingredient ?knife) 
			(not (spread_on_top ?bread))   
		) 
		:effect (and 
			(not (contains_ingredient ?knife))   
			(spread_on_top ?bread) 
		) 
	)
)