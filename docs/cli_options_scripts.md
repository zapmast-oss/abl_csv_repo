# CLI Options (scripts)

## scripts/build_story_menu_for_week.py
```
usage: build_story_menu_for_week.py [-h] [--min-priority {A,B,C}] week_label

Build a weekly story menu CSV from story_dictionary.csv.

positional arguments:
  week_label            Week label used in output filename, e.g. 1981_week_07

options:
  -h, --help            show this help message and exit
  --min-priority {A,B,C}
                        Minimum priority to include (default: A)
```

## scripts/eval_story_triggers_for_week.py
```
usage: eval_story_triggers_for_week.py [-h] [--min-priority {A,B,C}]
                                       week_label

Evaluate Pythagorean story triggers for a given week label and emit
story_candidates_<week_label>.csv

positional arguments:
  week_label            Week label, e.g. 1981_week_05

options:
  -h, --help            show this help message and exit
  --min-priority {A,B,C}
                        Minimum story priority to consider (default: A)
```
