(define (domain skillwrapper)

	(:requirements :adl :typing :equality :conditional-effects)

	(:types
		spreadable - object
		pickupable - object
		Jar - openable
		Table - location
		Jar - pickupable
		Bread - food
		RightGripper - arm
		location - object
		utensil - object
		LeftGripper - arm
		PeanutButter - spreadable
		Knife - utensil
		openable - object
		food - object
		Knife - pickupable
		arm - object
	)

	(:predicates
		(hand_empty ?arm - arm)
		(on_location ?pickupable - pickupable ?location - location)
		(contains ?openable - openable ?spreadable - spreadable)
		(is_graspable ?pickupable - pickupable ?arm - arm)
		(is_holding ?arm - arm ?pickupable - pickupable)
		(is_spread ?spreadable - spreadable ?food - food)
		(is_opened ?openable - openable)
	)

	(:action Pick
 		:parameters (?pickupable - pickupable ?arm - arm) 
		:precondition (and
			(is_graspable ?pickupable ?arm)
			(not (is_holding ?arm ?pickupable))
			(hand_empty ?arm)
		) 
		:effect (and
			(not (is_graspable ?pickupable ?arm))
			(is_holding ?arm ?pickupable)
			(not (hand_empty ?arm))
		)
	)

(:action Place
 		:parameters (?pickupable - pickupable ?arm - arm ?location - location) 
		:precondition (and
			(is_holding ?arm ?pickupable)
			(not (on_location ?pickupable ?location))
			(not (hand_empty ?arm))
	) 
		:effect (and
			(not (is_holding ?arm ?pickupable))
			(on_location ?pickupable ?location)
			(is_graspable ?pickupable ?arm)
			(hand_empty ?arm)
		)
	)

(:action Spread
 		:parameters (?utensil - utensil ?arm - arm ?spreadable - spreadable ?food - food) 
		:precondition (and
			(is_holding ?arm ?utensil)
			(contains ?utensil ?spreadable)
			(not (is_spread ?spreadable ?food))
	) 
		:effect (and
			(not (contains ?utensil ?spreadable))
			(is_spread ?spreadable ?food)
		)
	)

(:action Open
 	:parameters (?openable - openable ?arm1 - arm ?arm2 - arm) 
		:precondition (and
			(not (is_opened ?openable))
			(is_holding ?arm1 ?openable)
			(hand_empty ?arm2)
	) 
		:effect (and
			(is_opened ?openable)
		)
	)

(:action Scoop
 	:parameters (?utensil - utensil ?arm1 - arm ?arm2 - arm ?jar - openable ?ingredient - spreadable) 
		:precondition (and
			(is_holding ?arm1 ?utensil)
			(is_holding ?arm2 ?jar)
			(is_opened ?jar)
			(contains ?jar ?ingredient)
	) 
		:effect (and
			(contains ?utensil ?ingredient)
		)
	)
)