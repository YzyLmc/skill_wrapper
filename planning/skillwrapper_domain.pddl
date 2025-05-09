(define (domain skillwrapper)

	(:requirements :adl)

	(:types
		Door - door
		Robot - robot
		Table - location
		Cabinet - openable
		Door - location
		WhiteBoard - location
		location - object
		openable - object
		WhiteBoard - erasable
		Table - receptacle
		robot - object
		pickupable - object
		Eraser - pickupable
		eraser - object
		Cabinet - location
		Eraser - eraser
		erasable - object
		receptacle - object
		door - object
	)

	(:predicates
		(is_graspable ?robot - robot ?pickupable - pickupable)
		(is_properly_aligned ?robot - robot ?target - target)
		(is_within_reachable_zone ?robot - robot ?pickupable - pickupable)
		(has_proper_tool_attachment ?robot - robot ?tool - tool)
		(is_stably_positioned ?robot - robot ?pickupable - pickupable)
		(has_proper_contact ?robot - robot ?target - target)
		(is_within_grasping_orientation ?robot - robot ?pickupable - pickupable)
	)

	(:action a6 
:parameters ( ?pickupable_p0 - pickupable  ?pickupable_p1 - pickupable  ?robot_p2 - robot )
:precondition (and 
	(not (= ?pickupable_p0 ?pickupable_p1))
	(robot_pickupable_is_graspable ?robot_p2 ?pickupable_p0)
	(robot_pickupable_is_graspable ?robot_p2 ?pickupable_p1)
) 
:effect (and 
  ) 
)


(:action a8 
:parameters ( ?pickupable_p0 - pickupable  ?pickupable_p2 - pickupable  ?robot_p1 - robot )
:precondition (and 
	(not (= ?pickupable_p0 ?pickupable_p2))
	(robot_pickupable_is_graspable ?robot_p1 ?pickupable_p0)
	(robot_pickupable_is_graspable ?robot_p1 ?pickupable_p2)
	(not (robot_pickupable_is_stably_positioned ?robot_p1 ?pickupable_p0))
	(not (robot_pickupable_is_within_reachable_zone ?robot_p1 ?pickupable_p2))
) 
:effect (and 
  ) 
)

)