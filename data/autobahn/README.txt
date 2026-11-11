Munich raw data (not included in this repository)

Download the 1-minute loop-detector files of Die Autobahn GmbH des Bundes (A8-A99-A9 motorway
triangle) from the SCUDO Open Data Platform:

  https://opendataplatform.scudo-project.eu/?uc=traffic-management&cat=05-traffic-conditions

and put them in this folder (data/autobahn/), or set the environment variable TTA_DATA_DIR to
the folder that holds them.

Needed periods:
  2024-01-01_bis_2025-01-01   evaluation (held out for every model)
  2025-01-01_bis_2026-01-01   training of all learned models

Expected file names, one file per lane unit and period:
  FG1_Kurz_<unit>_agg1min_<period>.csv
  e.g. FG1_Kurz_69379081_DE1_agg1min_2024-01-01_bis_2025-01-01.csv

Format: semicolon-separated, one row per minute, columns
  t_start;q_kfz_h;q_lkw_h;q_pkw_h;v_kfz;v_lkw;v_pkw;b;t_netto;s_kfz

Detailed readme file is on the website along with the data.
