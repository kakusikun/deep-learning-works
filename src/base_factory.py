from abc import ABC, abstractmethod, abstractclassmethod

class BaseFactory(ABC):
    
    @classmethod
    def get_products(cls):
        assert hasattr(cls, 'products')
        try:
            return list(cls.products.keys())
        except:
            return cls.products
            
    @classmethod
    def produce(cls, product_name):
        if product_name not in cls.products:
            raise KeyError
        else:
            return cls.products[product_name]