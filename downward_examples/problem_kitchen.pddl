(define (problem organize_kitchen)
    (:domain kitchen_tasks)
    (:objects
        kitchen table fridge_location drawer_location microwave_location - location
        apple banana pie - item
        fridge drawer microwave - storage
    )
    (:init
        (hand-empty)
        (at kitchen)
        (storage-at fridge fridge_location)
        (storage-at drawer drawer_location)
        (storage-at microwave microwave_location)
        (item-at apple table)
        (item-at banana table)
        (item-at pie table)
        (is-closed fridge)
        (is-closed drawer)
        (is-closed microwave)
    )
    (:goal (and
        (in-storage apple fridge)
        (in-storage banana drawer)
        (in-storage pie microwave)
        (is-closed fridge)
        (is-closed drawer)
        (is-closed microwave)
    ))
)