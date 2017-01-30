"""
Lauchpoint for the CALL scraper and data file generator

"""
import configparser
import os
import sys
import logging
import multiprocessing

import CALL

logging.basicConfig(level=logging.ERROR)

if __name__ == '__main__':
    data_dir = ''
    db_file = '/call.db'
    settings_file = '/settings.ini'

    # fix windows specifics
    if sys.platform.startswith('win'):
        multiprocessing.freeze_support()
        # fix the terminal so we can print unicode
        import win_unicode_console
        win_unicode_console.enable()

        data_dir = os.environ['USERPROFILE'] + '\Documents\CALL'
        db_file = '\call.db'
        settings_file = '\settings.ini'
    else:
        data_dir = os.environ['HOME'] + '/CALL'

    # setup the data directory
    try:
        os.mkdir(data_dir)
        # nothing thrown, continue to set up the data dir
        logging.info("Copying call.db")
        with open('call.db', 'rb') as infile:
            with open(data_dir + db_file, 'wb') as outfile:
                outfile.write(infile.read())

        with open('settings.ini', 'r') as infile:
            with open(data_dir + settings_file, 'w') as outfile:
                outfile.write(infile.read())

    except FileExistsError:
        # data dir already exists
        pass

    logging.info("Moving to data dir: ", data_dir)
    os.chdir(data_dir)

    # load the settings
    settings = configparser.ConfigParser()
    settings.read('settings.ini')

    service = settings['SETTINGS']['service']
    similar_threshold = float(settings['SETTINGS']['similar_threshold'])
    term_threshold = float(settings['SETTINGS']['term_threshold'])

    print("Welcome to CALL Listing Generator")
    print('*' * 50)
    print()
    print("Press [ENTER] for previous values displayed in []")
    service_input = input("Enter update URL [{}]: ".format(service))
    if service_input:
        service = service_input

    similar_input = input("Enter similarity threshold [{}]: ".format(similar_threshold))
    if similar_input and float(similar_input) > 0.0:
        similar_threshold = float(similar_input)

    term_input = input("Enter term weight threshold [{}]: ".format(term_threshold))
    if term_input and float(term_input) > 0.0:
        term_threshold = float(term_input)

    # call the main function, on success, update all options
    if CALL.main(service=service, similar_threshold=similar_threshold, term_threshold=term_threshold):
        settings['SETTINGS']['service'] = service
        settings['SETTINGS']['similar_threshold'] = str(similar_threshold)
        settings['SETTINGS']['term_threshold'] = str(term_threshold)
        with open('settings.ini', 'w') as outfile:
            settings.write(outfile)

        print("Great success! Goodbye...")



