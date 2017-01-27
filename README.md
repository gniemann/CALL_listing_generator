These are the scripts which produce the publications JSON file:

The 'CALL' package contains all the scripts. run_scaper.py simply is a wrapper around this package. 

The 'alembic' files are for setting up the SQLite database required by the script. Running 'alembic upgrade head' creates an empty SQLite database. 'python3 run_scraper.py' then runs the scraper and fills it to the most current state. The datafile is output to updates.json. 

requirements.txt contains a list of the package dependencies. 