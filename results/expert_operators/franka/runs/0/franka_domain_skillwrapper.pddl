(define (domain franka_skillwrapper)

	(:requirements :adl :typing :equality :conditional-effects)

	(:types
		Teapot - pickupable
		Mug - container
		pickupable - object
		Sponge - pickupable
		Bowl - pickupable
		robot - object
		Plate - plate
		plate - object
		Robot - robot
		Teapot - pourable
		pourable - object
		container - object
		Sponge - sponge
		sponge - object
	)

	(:predicates
		(hand_empty ?robot - robot)
		(is_holding ?robot - robot ?pickupable - pickupable)
		(is_clean ?plate - plate)
		(is_on_top ?pickupable - pickupable ?plate - plate)
		(plate_empty ?plate - plate)
		(container_filled ?container - container)
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
			(not (container_filled ?container)) 
		) 
		:effect (and 
			(container_filled ?container) 
		)   
	)

(:action Stack
 		:parameters (?robot - robot ?pickupable - pickupable ?plate - plate) 
		:precondition (and 
			(not (hand_empty ?robot)) 
			(is_holding ?robot ?pickupable) 
			(not (is_on_top ?pickupable ?plate)) 
			(plate_empty ?plate) 
		) 
		:effect (and 
			(hand_empty ?robot) 
			(not (is_holding ?robot ?pickupable)) 
			(is_on_top ?pickupable ?plate) 
			(not (plate_empty ?plate)) 
		)   
	)

(:action Wipe
 		:parameters (?robot - robot ?sponge - sponge ?plate - plate) 
		:precondition (and 
			(is_holding ?robot ?sponge) 
			(not (is_clean ?plate)) 
			(plate_empty ?plate) 
		) 
		:effect (and 
			(is_clean ?plate) 
		)   
	)
)