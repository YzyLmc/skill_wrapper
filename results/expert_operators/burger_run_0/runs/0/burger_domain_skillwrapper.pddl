(define (domain burger_skillwrapper)

	(:requirements :adl :typing :equality :conditional-effects)

	(:types
		CuttingBoard - station
		BottomBun - pickupable
		Robot - robot
		Stove - station
		Lettuce - pickupable
		cuttable - object
		TopBun - pickupable
		cooker - object
		cuttingboard - object
		Stove - cooker
		CuttingBoard - cuttingboard
		robot - object
		pickupable - object
		Patty - pickupable
		Patty - cookable
		station - object
		Lettuce - cuttable
		cookable - object
	)

	(:predicates
		(hand_empty )
		(is_holding ?pickupable - pickupable)
		(obj_free ?pickupable - pickupable)
		(station_free ?station - station)
		(is_on_top ?pickupable - pickupable ?pickupable - pickupable)
		(is_on_station ?pickupable - pickupable ?station - station)
		(is_cut ?cuttable - cuttable)
		(is_cooked ?cookable - cookable)
	)

	(:action Pick
 		:parameters (?robot - robot ?pickupable - pickupable) 
		:precondition (and 
			(hand_empty) 
			(obj_free ?pickupable) 
			(not (is_holding ?pickupable))
		) 
		:effect (and 
			(not (hand_empty)) 
			(not (obj_free ?pickupable)) 
			(is_holding ?pickupable) 
		)
	)

(:action PickFromStack
 		:parameters (?robot - robot ?top - pickupable ?bot - pickupable) 
		:precondition (and 
			(hand_empty) 
			(not (is_holding ?top)) 
			(obj_free ?top) 
			(is_on_top ?top ?bot)
		) 
		:effect (and 
			(not (hand_empty)) 
			(is_holding ?top) 
			(obj_free ?bot) 
			(not (obj_free ?top)) 
			(not (is_on_top ?top ?bot)) 
		)
	)

(:action PickFromStation
 		:parameters (?robot - robot ?top - pickupable ?bot - station) 
		:precondition (and 
			(hand_empty) 
			(not (is_holding ?top)) 
			(obj_free ?top) 
			(is_on_station ?top ?bot)
		) 
		:effect (and 
			(not (hand_empty)) 
			(is_holding ?top) 
			(station_free ?bot) 
			(not (obj_free ?top)) 
			(not (is_on_station ?top ?bot)) 
		)
	)

(:action Place
 		:parameters (?robot - robot ?pickupable - pickupable ?station - station) 
		:precondition (and 
			(is_holding ?pickupable)   
			(station_free ?station) 
			(not (obj_free ?pickupable))
		) 
		:effect (and   
			(not (is_holding ?pickupable))   
			(obj_free ?pickupable) 
			(hand_empty)   
			(not (station_free ?station)) 
			(is_on_station ?pickupable ?station) 
		)
	)

(:action Stack
 		:parameters (?robot - robot ?top - pickupable ?bot - pickupable) 
		:precondition (and 
			(is_holding ?top) 
			(obj_free ?bot) 
			(not (obj_free ?top))
		) 
		:effect (and   
			(not (is_holding ?top)) 
			(hand_empty) 
			(obj_free ?top) 
			(not (obj_free ?bot)) 
			(is_on_top ?top ?bot) 
		)
	)

(:action Cut
 	:parameters (?robot - robot ?cuttable - cuttable ?board - cuttingboard) 
		:precondition (and 
			(hand_empty) 
			(obj_free ?cuttable) 
			(is_on_station ?cuttable ?board) 
			(not (is_cut ?cuttable))
		) 
		:effect (and 
			(is_cut ?cuttable) 
		)
	)

(:action Cook
 	:parameters (?robot - robot ?cookable - cookable ?cooker - cooker) 
		:precondition (and 
			(hand_empty) 
			(obj_free ?cookable) 
			(is_on_station ?cookable ?cooker) 
			(not (is_cooked ?cookable))
		) 
		:effect (and 
			(is_cooked ?cookable) 
		)
	)
)