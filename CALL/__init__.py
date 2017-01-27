"""
The scrape function is the entry point into the scraper
This package exposes a main() function, which serves only
to call the scraper.scrape() func.

The output of this is the updates.json file
"""


from CALL import scraper

def main():
    scraper.scrape()
