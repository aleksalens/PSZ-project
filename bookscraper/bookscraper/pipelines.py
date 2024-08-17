# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
from itemadapter import ItemAdapter


class BookscraperPipeline:
    def process_item(self, item, spider):
        
        adapter = ItemAdapter(item)
        
        field_names = adapter.field_names()
        for field_name in field_names:
            value = adapter.get(field_name)           
            if isinstance(value, str):
                adapter[field_name] = value.lstrip('\n\r').rstrip('\n\r').strip()

        
        price_keys = ['cena']
        for price_key in price_keys:
           value = adapter.get(price_key)
           if value is not None:
               value = value.replace(".", "").replace(",", ".")
           else:
               value = 0
           adapter[price_key] = float(value)    
        
        number_keys = ['broj_strana', 'godina_izdanja']
        for number_key in number_keys:
           value = adapter.get(number_key)
           if value is not None:
               value = value.replace(".", "")
           else:
               value = 0
           adapter[number_key] = int(value)
         
        return item

import mysql.connector

class SaveToDatabasePipeline:

    def __init__(self):
        self.conn = mysql.connector.connect(
            host = 'localhost',
            user = 'root',
            password = '****', # Your SQL password here
            database = 'books',
            charset='utf8mb4'
        )
   
        self.cur = self.conn.cursor()

        self.cur.execute(
            """
            CREATE TABLE IF NOT EXISTS books (
                id int NOT NULL auto_increment,
                naslov VARCHAR(255),
                autor VARCHAR(255),
                kategorija VARCHAR(255),
                izdavac VARCHAR(255),
                godina_izdanja INTEGER,
                broj_strana INTEGER,
                povez VARCHAR(255),
                format VARCHAR(255),
                opis TEXT,
                cena DECIMAL,
                PRIMARY KEY (id)
            )
            """)

    def process_item(self, item, spider):
        self.cur.execute("""
            INSERT INTO books (
                naslov,
                autor,
                kategorija,
                izdavac,
                godina_izdanja,
                broj_strana,
                povez,
                format,
                opis,
                cena
            ) 
            VALUES (
                %s,
                %s,
                %s,
                %s,
                %s,
                %s,
                %s,
                %s,
                %s,
                %s          
            )
        """, (
            item["naslov"],
            item["autor"],
            item["kategorija"],
            item["izdavac"],
            item["godina_izdanja"],
            item["broj_strana"],
            item["povez"],
            item["format"],
            str(item["opis"]),
            item["cena"]
        ))
        
        self.conn.commit()
        return item



    def close_spider(self, spider):
        self.conn.close()
        self.cur.close()


