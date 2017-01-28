"""
The scrape function is the entry point into the scraper
This package exposes a main() function, which serves only
to call the scraper.scrape() func.

The output of this is the updates.json file
"""
import sys

from CALL import scraper

def main(*args, **kwargs):
    try:
        scraper.scrape(*args, **kwargs)
        return True
    except:
        print("Oops, something went wrong: ", sys.exc_info()[0])
        return False
