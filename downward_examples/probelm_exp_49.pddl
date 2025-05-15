(define (problem move-items)
    (:domain exp_49)
    
    ;; Define the objects in the problem
    (:objects
        Vase TissueBox Bowl - object
        Sofa CoffeeTable DiningTable - location
    )
    
    ;; Define the initial state
    (:init
        ;; Locations of items
        (isat Vase CoffeeTable)
        (isat TissueBox Sofa)
        (isat Bowl DiningTable)

        ;; Robot's state
        (hasemptyhand)
        (isclose Sofa)

        ;; Items are initially unoccupied and loose
        (isloose Vase)
        (isloose TissueBox)
        (isloose Bowl)
        (isunoccupied CoffeeTable)
        (isunoccupied DiningTable)
    )
    
    ;; Define the goal state
    (:goal
        (and
            (isat Vase Sofa)
            (isat TissueBox Sofa)
            (isat Bowl Sofa)
        )
    )
)