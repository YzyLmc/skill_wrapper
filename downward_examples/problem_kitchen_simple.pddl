(define (problem organize_kitchen)
    (:domain kitchen_tasks)
    (:objects
        kitchen table apple_location banana_location pie_location - location
        apple banana pie - item
    )
    (:init
        (hand-empty)
        (at kitchen)
        (item-at apple table)
        (item-at banana table)
        (item-at pie table)
    )
    (:goal (and
        (item-at apple apple_location)
        (item-at banana banana_location)
        (item-at pie pie_location)
    ))
)