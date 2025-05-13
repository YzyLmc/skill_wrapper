(define (domain skillwrapper)

	(:requirements :adl :typing :equality :conditional-effects)

	(:types
		spreadable - object
		PeanutButter - spreadable
		utensil - object
		Table - location
		Jar - pickupable
		Bread - food
		receptacle - object
		food - object
		openable - object
		Knife - utensil
		hand - object
		Gripper - hand
		Shelf - location
		location - object
		utensil - pickupable
		pickupable - object
		Cup - receptacle
		Jar - openable
	)

	(:predicates
		(hand_empty ?hand - hand)
		(on_location ?pickupable - pickupable ?location - location)
		(contains ?pickupable - pickupable ?spreadable - spreadable)
		(is_graspable ?pickupable - pickupable ?hand - hand)
		(is_holding ?hand - hand ?pickupable - pickupable)
		(is_spread ?spreadable - spreadable ?food - food)
		(is_opened ?openable - openable)
	)

	(:action Pick
 		:parameters (?pickupable - pickupable ?hand - hand) 
		:precondition (and
			(is_graspable ?pickupable ?hand)
			(not (is_holding ?hand ?pickupable))
			(hand_empty ?hand)
		) 
		:effect (and
			(not (is_graspable ?pickupable ?hand))
			(is_holding ?hand ?pickupable)
			(not (hand_empty ?hand))
		)
	)

(:action Place
 		:parameters (?pickupable - pickupable ?hand - hand ?location - location) 
		:precondition (and
			(is_holding ?hand ?pickupable)
			(not (on_location ?pickupable ?location))
			(not (hand_empty ?hand))
	) 
		:effect (and
			(not (is_holding ?hand ?pickupable))
			(on_location ?pickupable ?location)
			(is_graspable ?pickupable ?hand)
			(hand_empty ?hand)
		)
	)

(:action Spread
 		:parameters (?utensil - utensil ?hand - hand ?spreadable - spreadable ?food - food) 
		:precondition (and
			(is_holding ?hand ?utensil)
			(contains ?utensil ?spreadable)
			(not (is_spread ?spreadable ?food))
	) 
		:effect (and
			(not (contains ?utensil ?spreadable))
			(is_spread ?spreadable ?food)
		)
	)

(:action Open
 	:parameters (?openable - openable ?arm1 - hand ?arm2 - hand) 
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
 	:parameters (?utensil - utensil ?arm1 - hand ?arm2 - hand ?jar - openable ?ingredient - spreadable) 
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