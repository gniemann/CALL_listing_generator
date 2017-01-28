"""
Database models
"""

from datetime import datetime
from contextlib import contextmanager

from sqlalchemy import Column, Table, Integer, ForeignKey, String, Date, Text, Float
from sqlalchemy.orm import relationship
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

engine = create_engine('sqlite:///call.db', echo=False)
Session = sessionmaker(bind=engine)

@contextmanager
def open_session():
    try:
        session = Session()
        yield session
        session.commit()
        print('Changes committed')
    except:
        session.rollback()
        raise
    finally:
        session.close()


BaseModel = declarative_base()

# Define similiar publiciations table
similar_publications = Table('SimilarPublications', BaseModel.metadata,
                                 Column('left_id', Integer, ForeignKey('publication.id'), primary_key=True),
                                 Column('right_id', Integer, ForeignKey('publication.id'), primary_key=True))

class Publication(BaseModel):
    __tablename__ = 'publication'

    id = Column(Integer, primary_key=True)
    title = Column(String(255))
    abstract = Column(Text)
    date_published = Column(Date)
    image_url = Column(String(255))
    publication_url = Column(String(255))
    type_id = Column(Integer, ForeignKey('publication_type.id'))
    terms = relationship('SearchTerms', back_populates='publication')
    similar = relationship('Publication', secondary=similar_publications,
                           primaryjoin=similar_publications.c.left_id == id,
                           secondaryjoin=similar_publications.c.right_id == id,
                            foreign_keys=[similar_publications.c.left_id, similar_publications.c.right_id])

    def __init__(self, **kwargs):
        self.last_modified = datetime.utcnow()

        for kw, arg in kwargs.items():
            self.__setattr__(kw, arg)

    def to_dict(self, threshold):
        return {
            'id': self.id,
            'title': self.title,
            'abstract': self.abstract,
            'date_published': str(self.date_published),
            'image_url': self.image_url,
            'publication_url': self.publication_url,
            'type': self.type.type,
            'terms': ' '.join([t.term.term for t in self.terms if t.weight > threshold]),
            'similar': [d.id for d in self.similar]
        }

    def __repr__(self):
        return '<Publication {} - {}'.format(self.id, self.title)

class PublicationType(BaseModel):
    __tablename__ = 'publication_type'

    id = Column(Integer, primary_key=True)
    type = Column(String(255))
    publications = relationship('Publication', backref='type')

class SearchTerm(BaseModel):
    __tablename__ = 'search_term'

    id = Column(Integer, primary_key=True)
    term = Column(String(50))
    publications = relationship('SearchTerms', back_populates='term')

class SearchTerms(BaseModel):
    __tablename__ = 'search_terms'
    pub_id = Column(Integer, ForeignKey('publication.id'), primary_key=True)
    term_id = Column(Integer, ForeignKey('search_term.id'), primary_key=True)
    occurs = Column(Integer)
    weight = Column(Float)
    term = relationship('SearchTerm', back_populates='publications')
    publication = relationship('Publication', back_populates='terms')

