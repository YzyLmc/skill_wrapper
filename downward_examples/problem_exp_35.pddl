(define (problem move-items)
    (:domain exr_35)
    
    ;; Define the objects in the problem
    (:objects
        Vase TissueBox Bowl - object
        Sofa CoffeeTable DiningTable - location
    )
    
    ;; Define the initial state
    (:init
        ;; Items are initially at certain locations
        (isat Vase CoffeeTable)
        (isat TissueBox Sofa)
        (isat Bowl DiningTable)
        
        ;; The robot's initial state is near the Sofa
        (isnear Sofa)

        ;; All items are movable
        (ismovable Vase)
        (ismovable TissueBox)
        (ismovable Bowl)

        ;; All locations can hold any object
        (canhold Vase Sofa)
        (canhold Vase CoffeeTable)
        (canhold Vase DiningTable)
        (canhold TissueBox Sofa)
        (canhold TissueBox CoffeeTable)
        (canhold TissueBox DiningTable)
        (canhold Bowl Sofa)
        (canhold Bowl CoffeeTable)
        (canhold Bowl DiningTable)
        
        ;; All locations can support any object
        (cansupport Vase Sofa)
        (cansupport Vase CoffeeTable)
        (cansupport Vase DiningTable)
        (cansupport TissueBox Sofa)
        (cansupport TissueBox CoffeeTable)
        (cansupport TissueBox DiningTable)
        (cansupport Bowl Sofa)
        (cansupport Bowl CoffeeTable)
        (cansupport Bowl DiningTable)

        ;; All locations are initially accessible
        (isaccessible Sofa)
        (isaccessible CoffeeTable)
        (isaccessible DiningTable)

        ;; The robot has empty hands initially
        (isempty Sofa)
        (isempty CoffeeTable)
        (isempty DiningTable)
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