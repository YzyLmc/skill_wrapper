(define (domain burger_skillwrapper)

	(:requirements :adl :typing :equality :conditional-effects)

	(:types
		cookable - object
		robot - object
		TopBun - pickupable
		station - object
		cooker - object
		Robot - robot
		Patty - pickupable
		Lettuce - cuttable
		BottomBun - pickupable
		cuttable - object
		CuttingBoard - station
		CuttingBoard - cuttingboard
		cuttingboard - object
		pickupable - object
		Patty - cookable
		Stove - station
		Stove - cooker
		Lettuce - pickupable
	)

	(:predicates
		(iscooked ?cookable - cookable)
		(iscut ?cuttable - cuttable)
		(nothing ?robot - robot)
		(station_empty ?station - station)
		(item_on ?pickupable - pickupable ?station - station)
		(clear ?pickupable - pickupable)
		(atop ?pickupable - pickupable ?pickupable - pickupable)
		(has_item ?robot - robot ?pickupable - pickupable)
	)

	(:action Pick_120 
:parameters ( ?pickupable_p2 - pickupable  ?pickupable_p3 - pickupable  ?robot_p0 - robot )
:precondition (and 
	(not (= ?pickupable_p2 ?pickupable_p3))
	(clear ?pickupable_p2 )
	(nothing ?robot_p0 )
	(not (has_item ?robot_p0 ?pickupable_p2))
	(not (has_item ?robot_p0 ?pickupable_p3))
) 
:effect (and 
 	(has_item ?robot_p0 ?pickupable_p2) 
	(not (clear ?pickupable_p2 ))
	(not (nothing ?robot_p0 ))
 ) 
)


(:action Pick_121 
:parameters ( ?pickupable_p1 - pickupable  ?pickupable_p2 - pickupable  ?pickupable_p4 - pickupable  ?robot_p0 - robot )
:precondition (and 
	(not (= ?pickupable_p1 ?pickupable_p2))
	(not (= ?pickupable_p1 ?pickupable_p4))
	(not (= ?pickupable_p2 ?pickupable_p4))
	(atop ?pickupable_p2 ?pickupable_p4)
	(nothing ?robot_p0 )
	(not (atop ?pickupable_p1 ?pickupable_p2))
	(not (clear ?pickupable_p4 ))
	(not (has_item ?robot_p0 ?pickupable_p1))
	(not (has_item ?robot_p0 ?pickupable_p2))
	(not (has_item ?robot_p0 ?pickupable_p4))
) 
:effect (and 
 	(clear ?pickupable_p4 ) 
	(has_item ?robot_p0 ?pickupable_p2) 
	(not (atop ?pickupable_p2 ?pickupable_p4))
	(not (clear ?pickupable_p2 ))
	(not (nothing ?robot_p0 ))
 ) 
)


(:action Pick_122 
:parameters ( ?pickupable_p1 - pickupable  ?pickupable_p2 - pickupable  ?pickupable_p3 - pickupable  ?pickupable_p4 - pickupable  ?robot_p0 - robot  ?station_p5 - station )
:precondition (and 
	(not (= ?pickupable_p1 ?pickupable_p2))
	(not (= ?pickupable_p1 ?pickupable_p3))
	(not (= ?pickupable_p1 ?pickupable_p4))
	(not (= ?pickupable_p2 ?pickupable_p3))
	(not (= ?pickupable_p2 ?pickupable_p4))
	(not (= ?pickupable_p3 ?pickupable_p4))
	(clear ?pickupable_p2 )
	(item_on ?pickupable_p2 ?station_p5)
	(nothing ?robot_p0 )
	(not (atop ?pickupable_p2 ?pickupable_p4))
	(not (has_item ?robot_p0 ?pickupable_p1))
	(not (has_item ?robot_p0 ?pickupable_p2))
	(not (has_item ?robot_p0 ?pickupable_p3))
	(not (station_empty ?station_p5 ))
) 
:effect (and 
 	(has_item ?robot_p0 ?pickupable_p2) 
	(station_empty ?station_p5 ) 
	(not (clear ?pickupable_p2 ))
	(not (item_on ?pickupable_p2 ?station_p5))
	(not (nothing ?robot_p0 ))
 ) 
)


(:action Place_123 
:parameters ( ?pickupable_p2 - pickupable  ?pickupable_p4 - pickupable  ?robot_p0 - robot  ?station_p5 - station )
:precondition (and 
	(not (= ?pickupable_p2 ?pickupable_p4))
	(has_item ?robot_p0 ?pickupable_p2)
	(station_empty ?station_p5 )
	(not (clear ?pickupable_p2 ))
	(not (has_item ?robot_p0 ?pickupable_p4))
	(not (item_on ?pickupable_p2 ?station_p5))
	(not (nothing ?robot_p0 ))
) 
:effect (and 
 	(clear ?pickupable_p2 ) 
	(item_on ?pickupable_p2 ?station_p5) 
	(nothing ?robot_p0 ) 
	(not (has_item ?robot_p0 ?pickupable_p2))
	(not (station_empty ?station_p5 ))
 ) 
)


(:action Stack_124 
:parameters ( ?pickupable_p2 - pickupable  ?pickupable_p4 - pickupable  ?robot_p0 - robot )
:precondition (and 
	(not (= ?pickupable_p2 ?pickupable_p4))
	(clear ?pickupable_p4 )
	(has_item ?robot_p0 ?pickupable_p2)
	(not (atop ?pickupable_p2 ?pickupable_p4))
	(not (has_item ?robot_p0 ?pickupable_p4))
	(not (nothing ?robot_p0 ))
) 
:effect (and 
 	(atop ?pickupable_p2 ?pickupable_p4) 
	(clear ?pickupable_p2 ) 
	(nothing ?robot_p0 ) 
	(not (clear ?pickupable_p4 ))
	(not (has_item ?robot_p0 ?pickupable_p2))
 ) 
)


(:action Cut_125 
:parameters ( ?pickupable_p1 - pickupable  ?pickupable_p2 - pickupable  ?pickupable_p3 - pickupable  ?pickupable_p4 - pickupable  ?robot_p0 - robot  ?station_p0 - station )
:precondition (and 
	(not (= ?pickupable_p1 ?pickupable_p2))
	(not (= ?pickupable_p1 ?pickupable_p3))
	(not (= ?pickupable_p1 ?pickupable_p4))
	(not (= ?pickupable_p2 ?pickupable_p3))
	(not (= ?pickupable_p2 ?pickupable_p4))
	(not (= ?pickupable_p3 ?pickupable_p4))
	(clear ?pickupable_p3 )
	(item_on ?pickupable_p3 ?station_p0)
	(nothing ?robot_p0 )
	(not (atop ?pickupable_p3 ?pickupable_p1))
	(not (has_item ?robot_p0 ?pickupable_p2))
	(not (has_item ?robot_p0 ?pickupable_p3))
	(not (has_item ?robot_p0 ?pickupable_p4))
	(not (iscut ?pickupable_p3 ))
) 
:effect (and 
 	(iscut ?pickupable_p3 ) 
 ) 
)


(:action Cook_126 
:parameters ( ?pickupable_p1 - pickupable  ?pickupable_p2 - pickupable  ?pickupable_p4 - pickupable  ?robot_p0 - robot  ?station_p5 - station )
:precondition (and 
	(not (= ?pickupable_p1 ?pickupable_p2))
	(not (= ?pickupable_p1 ?pickupable_p4))
	(not (= ?pickupable_p2 ?pickupable_p4))
	(clear ?pickupable_p2 )
	(item_on ?pickupable_p2 ?station_p5)
	(nothing ?robot_p0 )
	(not (atop ?pickupable_p1 ?pickupable_p2))
	(not (atop ?pickupable_p2 ?pickupable_p4))
	(not (has_item ?robot_p0 ?pickupable_p2))
	(not (has_item ?robot_p0 ?pickupable_p4))
	(not (iscooked ?pickupable_p2 ))
) 
:effect (and 
 	(iscooked ?pickupable_p2 ) 
 ) 
)

)