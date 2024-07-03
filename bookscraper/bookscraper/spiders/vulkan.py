import scrapy
from bookscraper.items import BookItem

class VulkanSpider(scrapy.Spider):
    name = "vulkan"
    allowed_domains = ["knjizare-vulkan.rs"]
    start_urls = ["https://www.knjizare-vulkan.rs/domace-knjige/page-1"]

    custom_settings = {
        'FEEDS': {
            'vulkan_books.json': {'format': 'json', 'overwrite': True},   
        }
    }
    
    def parse(self, response):
        books = response.css('div.wrapper-gridalt-view.item.product-item')

        for book in books:
            book_url = book.css('div.item-data div.img-wrapper a').attrib['href']        
            
            yield response.follow(book_url, callback=self.parse_book_page)        
        
            
        next_page = self.get_next_page_url(response)
        
        if next_page is not None:
            yield response.follow(next_page, callback = self.parse)
    
    
    def get_next_page_url(self, response):
        next_page_number = response.css('a.icon-caret-right::attr(href)').re_first(r'loadProductForPage\((\d+)\)')
        
        if next_page_number is not None:
            return response.url.split('/page-')[0]  + '/page-' + next_page_number
        else: 
            return None
            
          
             
    def parse_book_page(self, response):        
        book = BookItem()
        
        
        book['naslov'] = response.css('div.title h1 span::text').get()
        book['autor'] = response.xpath('//div[@class="item"]/span[@class="title"][text()="Autor :"]/following-sibling::span[@class="value"]/a/text()').get()
        book['kategorija'] = response.xpath('//div[@class="category"]/a/text()').get()
        book['izdavac'] = response.xpath('//div[@class="item"]/span[@class="title"][text()="Izdavač :"]/following-sibling::span[@class="value"]/a/text()').get()
        book['godina_izdanja'] = response.xpath('//tr[td="Godina"]/td[2]/text()').get()
        book['broj_strana'] = response.xpath('//tr[td="Strana"]/td[2]/text()').get()
        book['povez'] = response.xpath('//tr[td="Povez"]/td[2]/text()').get()
        book['format'] = response.xpath('//tr[td="Format"]/td[2]/text()').get()
        book['opis'] = response.xpath('//div[@class="description  read-more-text less"]/text()').get()
        book['cena'] = response.xpath('normalize-space(//span[@class="product-price-value value "])').get()
        
        yield book

