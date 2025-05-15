(define (problem move-items)
    (:domain exp_33)
    
    ;; Define the objects in the problem
    (:objects
        Vase TissueBox Bowl - object
        Sofa CoffeeTable DiningTable - location
    )
    
    ;; Define the initial state
    (:init
        ;; Locations of items
        (isstableplacement Vase CoffeeTable)
        (isstableplacement TissueBox Sofa)
        (isstableplacement Bowl DiningTable)

        ;; Robot's state
        (at Sofa)
        (near TissueBox)

        ;; All items can fit in all locations
        (canfit Vase Sofa)
        (canfit Vase CoffeeTable)
        (canfit Vase DiningTable)
        (canfit TissueBox Sofa)
        (canfit TissueBox CoffeeTable)
        (canfit TissueBox DiningTable)
        (canfit Bowl Sofa)
        (canfit Bowl CoffeeTable)
        (canfit Bowl DiningTable)

        ;; Items are fixed/stable in their initial locations
        (isfixed Vase)
        (isfixed TissueBox)
        (isfixed Bowl)
    )
    
    ;; Define the goal state
    (:goal
        (and
            (isstableplacement Vase Sofa)
            (isstableplacement TissueBox Sofa)
            (isstableplacement Bowl Sofa)
        )
    )
)
