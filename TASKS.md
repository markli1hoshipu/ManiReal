# RoboTwin Benchmark Tasks & Data Collection Plan

## Data Collection Target

Each task should have:
- **Clean**: 100 episodes (obstacle_density = 0)
- **Cluttered**: 10 densities (d6 through d15) x 10 episodes each = 100 episodes total

Total per task: 200 episodes (100 clean + 100 cluttered)

---

## Study (20 tasks)

| # | Task Description | Code File | Type |
|---|---|---|---|
| 1 | Empty the box | `empty_box` | Mid-range |
| 2 | Move the book onto the table | `move_book_onto_table` | Atomic |
| 3 | Move the cup to the right/left | `move_cup` | Atomic |
| 4 | Move the cup next to the book | `move_cup_next_to_book` | Atomic |
| 5 | Move the cup onto the table | `move_cup_onto_table` | Atomic |
| 6 | Move the cup, then put the pen in the cup | `move_cup_put_pen_in_cup` | Mid-range |
| 7 | Move the cups into the box | `move_cups_into_box` | Mid-range |
| 8 | Move the pen to the box | `move_pen_to_box` | Atomic |
| 9 | Move the seal next to box, put cup in box | `move_seal_cup_next_to_box` | Mid-range |
| 10 | Move the seal next to the box | `move_seal_next_to_box` | Atomic |
| 11 | Move the seal next to the pencup | `move_seal_next_to_pencup` | Atomic |
| 12 | Move the seal onto the book | `move_seal_onto_book` | Atomic |
| 13 | Move the seal onto the table | `move_seal_onto_table` | Atomic |
| 14 | Put the cup in the box | `put_cup_in_box` | Atomic |
| 15 | Put the cup on the coaster | `put_cup_on_coaster` | Atomic |
| 16 | Put the cup on the table | `put_cup_on_table` | Atomic |
| 17 | Put the glue in the box | `put_glue_in_box` | Atomic |
| 18 | Put the pen in the box | `put_pen_in_box` | Atomic |
| 19 | Put the pen in the pencup | `put_pen_in_pencup` | Atomic |
| 20 | Put the seal in the box | `put_seal_in_box` | Atomic |

## Office (20 tasks)

| # | Task Description | Code File | Type |
|---|---|---|---|
| 1 | Move the milktea to the right of the laptop | `put_milktea_next_to_laptop` | Atomic |
| 2 | Put the stapler in the open drawer | `put_stapler_in_drawer` | Atomic |
| 3 | Stack the book from file holder on top of the book on the table | `put_book_on_book` | Atomic |
| 4 | Put the phone on the holder | `put_phone_on_holder` | Atomic |
| 5 | Move the phone to beside the cube | `put_phone_next_to_cube` | Atomic |
| 6 | Put the mouse on the mousepad | `put_mouse_on_pad` | Atomic |
| 7 | Place the milktea on the lower shelf | `put_milktea_on_shelf` | Atomic |
| 8 | Pick the mouse from the open drawer and place it to the right of the stapler | `put_mouse_next_to_stapler` | Atomic |
| 9 | Open the drawer | `open_drawer` | Atomic |
| 10 | Close the drawer | `close_drawer` | Atomic |
| 11 | Move the book from the shelf to the lower level of the file holder | `put_book_in_fileholder` | Atomic |
| 12 | Place the rubiks cube to the left of the milktea | `put_rubikscube_next_to_milktea` | Atomic |
| 13 | Place the stapler next to the mouse | `put_stapler_next_to_mouse` | Atomic |
| 14 | Move rubiks cube to drawer | `put_rubikscube_in_drawer` | Atomic |
| 15 | Place the stapler on the book | `put_stapler_on_book` | Atomic |
| 16 | Open the drawer, place the stapler inside, close the drawer | `store_stapler_in_drawer` | Mid-range |
| 17 | Milktea to coaster, rubiks cube to shelf, notebook to book | `organize_table` | Mid-range |
| 18 | Mouse to pad, phone to holder, milktea to shelf | `set_up_table` | Mid-range |
| 19 | Open drawer, rubiks cube to shelf, close drawer | `store_rubikscube_on_shelf` | Mid-range |
| 20 | Mouse from drawer to pad, close drawer, book to file holder | `move_items_around` | Mid-range |

## HuggingFace Dataset

Repository: [Hoshipu/roboreal_data](https://huggingface.co/datasets/Hoshipu/roboreal_data)

```
study/
  {task_name}/
    clean/          # 100 episodes, density 0
    d6/             # 10 episodes, density 6
    d7/             # 10 episodes, density 7
    ...
    d15/            # 10 episodes, density 15
office/
  {task_name}/
    clean/
    d6/
    ...
    d15/
```
