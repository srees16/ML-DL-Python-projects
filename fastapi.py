'''FastAPI'''

# Path parameters vs. query parameters: path parameters are to be declared as part of the path to get the info, query parameters is for inputting the queries
'''
practice FastAPI:
    https://www.youtube.com/watch?v=-ykeT6kk4bk
	https://www.youtube.com/watch?v=7t2alSnE2-I
	https://www.youtube.com/watch?v=kyJNbSUz86w
    https://www.youtube.com/watch?v=StHtw2YvNFU

practive web scraping:
    scrape wikipedia tables
    scrape youtube vid titles, views, likes, dislikes

OAuth2:
    https://www.youtube.com/watch?v=6hTRw_HK3Ts
    https://www.youtube.com/watch?v=cXy4Fy1smX8 (https://www.youtube.com/playlist?list=PL0lYY7rL__yJKvrhuIae-SHY7bZm9Vasb)
    https://www.youtube.com/watch?v=xZnOoO3ImSY

# Build flames app
'''


from sklearn.datasets import make_classification
import uvicorn # ASGI
from fastapi import FastAPI, Path, HTTPException, status
from pydantic import BaseModel


class Item(BaseModel):
    
    name: str
    price: float
    brand: str = None

class UpdateItem(BaseModel):
    
    name: str = None
    price: float = None
    brand: str = None

x, y = make_classification(n_samples = 1000, n_features = 10, random_state = 8, shuffle = True)

# create app object
app = FastAPI()

@app.get('/')
def index():
    return {'message': 'Hello World'}

# path parameters
@app.get('/app/{name}')
def welcome(name: str):
    return 'Welcome to my page, Mr. {}'.format(name)


# query parameters
@app.get('/apps')
def welcome(name: str):
    return 'Welcome to my page, Mr. {}'.format(name)


inventory = {
    1: {
        'name': 'milk',
        'price': 30,
        'brand': 'heritage'
        }
    }


@app.get('/get_items')
def fetch_all_items():
    return inventory


@app.get('/get_item/{item_id}') # match path parameter in URI with arguement in below function
def fetch_item(item_id: int = Path(default=None, description='This is just a test path', ge=1, lt=100)): # int is type hint, can add constraints in Path
    return inventory[item_id]


# multiple path parameters
@app.get('/get_item/{item_id}/{item_name}')
def fetch_detail(item_id: int, item_name: str):
    return inventory[item_id][item_name]

#fetch_detail(2, 'brand')

# multiple query parameters
@app.get('/get_item_by_detail')
def fetch_details_queries(item_name: str):
    for item in inventory:
        if inventory[item]['brand'] == item_name:
            return inventory[item]
        elif inventory[item]['name'] == item_name:
            return inventory[item]
        elif inventory[item]['price'] == item_name:
            return inventory[item]

#fetch_details_queries('milk')
#fetch_details_queries('heritage')
#fetch_details_queries(30)

# post method
@app.post('/create/{item_id}')
def create(item_id: int, data: Item):
    if item_id in inventory:
        raise HTTPException(status_code = status.HTTP_208_ALREADY_REPORTED, detail='Item ID not found') # or status_code = 208
    inventory[item_id] = data
    return inventory[item_id]


# put method
@app.put('/update/{item_id}')
def update(item_id: int, item: UpdateItem):
    if item_id not in inventory:
        return {'Error': 'Item not present'}
    if item.name != None:
        inventory[item_id]['name'] = item.name
    if item.price != None:
        inventory[item_id]['price'] = item.price
    if item.brand != None:
        inventory[item_id]['brand'] = item.brand
    return inventory[item_id]

'''ite = {
  "name": "bread",
  "price": 6.44,
  "brand": "loafty"
}

update(2, ite)
'''

@app.delete('/delete_item')
def deletion(item_id: int):
    if item_id not in inventory:
        return {'msg': 'item not present'}
    else:
        del inventory[item_id]
    return {'msg': 'item {} deleted'.format(item_id)}


# run the API with uvicorn
if __name__ == '__main__':
    uvicorn.run(app, host = '127.0.0.2', port = 5000, debug=True)

#uvicorn main:app --reload