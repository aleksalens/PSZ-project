# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class BookItem(scrapy.Item):
    naslov = scrapy.Field()
    autor = scrapy.Field()
    kategorija = scrapy.Field()
    izdavac = scrapy.Field()
    godina_izdanja = scrapy.Field()
    broj_strana = scrapy.Field()
    povez = scrapy.Field()
    format = scrapy.Field()
    opis = scrapy.Field()
    cena = scrapy.Field()
    