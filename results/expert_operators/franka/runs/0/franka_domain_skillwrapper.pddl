(define (domain franka_skillwrapper)

	(:requirements :adl :typing :equality :conditional-effects)

	(:types
		Teapot - pourable
		pourable - object
		plate - object
		robot - object
		Sponge - sponge
		container - object
		Plate - plate
		Sponge - pickupable
		pickupable - object
		Mug - container
		Bowl - pickupable
		Teapot - pickupable
		Robot - robot
		sponge - object
	)

	(:predicates
		(hand_empty ?robot - robot)
		(is_holding ?robot - robot ?pickupable - pickupable)
		(is_clean ?plate - plate)
		(is_on_top ?pickupable - pickupable ?plate - plate)
		(is_empty ?container - container)
	)

	(:action Pick
 		:parameters (?robot - robot ?pickupable - pickupable) 
		:precondition (and 
			(hand_empty ?robot) 
			(not (is_holding ?robot ?pickupable)) 
		) 
		:effect (and 
			(not (hand_empty ?robot)) 
			(is_holding ?robot ?pickupable) 
		)   
	)

(:action Place
 		:parameters (?robot - robot ?pickupable - pickupable) 
		:precondition (and 
			(not (hand_empty ?robot)) 
			(is_holding ?robot ?pickupable) 
		) 
		:effect (and 
			(hand_empty ?robot) 
			(not (is_holding ?robot ?pickupable)) 
		)   
	)

(:action Pour
 		:parameters (?robot - robot ?pourable - pourable ?container - container) 
		:precondition (and 
			(is_holding ?robot ?pourable) 
			(is_empty ?container) 
		) 
		:effect (and 
			(not (is_empty ?container)) 
		)   
	)

(:action Stack
 		:parameters (?robot - robot ?pickupable - pickupable ?plate - plate) 
		:precondition (and 
			(not (hand_empty ?robot)) 
			(is_holding ?robot ?pickupable) 
			(not (is_on_top ?pickupable ?plate)) 
		) 
		:effect (and 
			(hand_empty ?robot) 
			(not (is_holding ?robot ?pickupable)) 
			(is_on_top ?pickupable ?plate) 
		)   
	)

(:action Wipe
 		:parameters (?robot - robot ?sponge - sponge ?plate - plate) 
		:precondition (and 
			(is_holding ?robot ?sponge) 
			(not (is_clean ?plate)) 
		) 
		:effect (and 
			(is_clean ?plate) 
		)   
	)
)