import json
from datetime import datetime

import requests
from lxml import html

from CALL import pub_analyzer
from CALL.models import open_session, Publication, SearchTerm, PublicationType


PUBLICATION_URL = 'http://usacac.army.mil/organizations/mccoe/call/publications'

def download_publications_page(pubs_url):
    pubs_request = requests.get(pubs_url)

    if pubs_request.status_code != 200:
        print("Request to {} returned error {}: {}".format(pubs_url, pubs_request.status_code,
                                                           pubs_request.reason))
        return None

    print("Page retrieval successful")
    return pubs_request.text

def fix_link(link):
    return link.replace('pubs_page_files', 'sites/default/files/covers')

def parse_page(page):
    pubs_page = html.fromstring(page)

    pubs_page.rewrite_links(fix_link, base_href='http://usacac.army.mil')

    pubs = []

    article = pubs_page.find_class('field-name-body')[0]
    publications = article.find_class('field-item')[0]

    types = [heading.text for heading in publications.findall('h2')]

    for section, type in zip(publications.find_class('view-pubs'), types):
        for block in section.find_class('field-content'):
            pub_block = block[0]
            pub = {}
            link = pub_block.find('a')
            pub['publication_url'] = link.get('href')
            pub['image_url'] = link.find('img').get('src')
            pub['title'] = link.find('strong').text

            meta_data = pub_block.find_class('pub-meta')[0]
            pub['date_published'] = datetime.strptime(meta_data[0].text, '%d %b %Y').date()

            description = pub_block.find_class('pub-desc')[0].text_content() or ''

            description = description.replace('\n', ' ')
            description = description.replace('\r', '')
            description = description.replace('  ', ' ')
            pub['abstract'] = description
            pub['type'] = type

            pubs.append(pub)

    next_button = pubs_page.find_class('pager-next')

    if next_button:
        print("Getting next page")
        next_link = next_button[0].find('a').get('href')

        pubs.extend(parse_page(download_publications_page(next_link)))

    return pubs

def update_database(scraped_pubs, session):
    new_pubs_added = []
    scraped_pubs.sort(key=lambda x: x['date_published'])
    for pub in scraped_pubs:
        type_obj = session.query(PublicationType).filter_by(type=pub['type']).one_or_none()
        if not type_obj:
            type_obj = PublicationType(type=pub['type'])
            session.add(type_obj)
        pub['type'] = type_obj

        old_pub = session.query(Publication).filter_by(title=pub['title']).one_or_none()
        if old_pub:
            print('Updating pub {}: {}'.format(old_pub.id, old_pub.title))
            for attr, value in pub.items():
                if attr in old_pub.__dict__ and old_pub.__dict__[attr] != value:
                    old_pub.__setattr__(attr, value)
                    print("{} updated to {}".format(attr, value))

        else:
            print('Adding new pub: {}'.format(pub['title']))
            new_pub = Publication(**pub)
            new_pubs_added.append(new_pub)
            session.add(new_pub)

    if new_pubs_added:
        print("Running the analyzer...")
        pub_analyzer.run_analyzer(new_pubs_added, session)


def generate_pubs_json(session):
    pubs = [p.to_dict(0.025) for p in session.query(Publication).all()]

    results = {
        'service': 'https://centerarmylessonslearned.herokuapp.com/updates',
        'messages': [],
        'publications': pubs
    }

    with open('publications.json', 'w') as file:
        json.dump(results, file, indent=2)

def scrape():
    scraped_pubs = parse_page(download_publications_page(PUBLICATION_URL))
    with open_session() as session:
        update_database(scraped_pubs, session)
        generate_pubs_json(session)


if __name__ == '__main__':
    scrape()

