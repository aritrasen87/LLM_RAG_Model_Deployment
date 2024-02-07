# pip install fastapi uvicorn

from fastapi import FastAPI
import uvicorn
import os
from dotenv import load_dotenv
from enum import Enum
from pydantic import BaseModel


load_dotenv()

app = FastAPI()

# @app.get("/hello")
# async def hello():
#     return 'Hello World'

@app.get("/hello/{name}")
async def hello(name:str):
    return f"Hello {name} "

models = {
    'LLMs' : ['OpenAI', 'Mistral'],
    'NLP' : ['Bert', 'RoBerta'],
    'ML' : ['Xgboost', 'Catboost']
}

# @app.get("/get_models/{usecase}")
# async def get_items(usecase:str):
#     return models.get(usecase)

### Validation:

class AvailableModel(str, Enum):
    LLMs = "LLMs"
    NLP = "NLP"
    ML = "ML"
    
@app.get("/get_models/{usecase}")
async def get_items(usecase: AvailableModel):
    return models.get(usecase)

class Item(BaseModel):
    name: str
    description: str | None = None
    price: float
    tax: float | None = None

@app.post("/items/")
async def create_item(item: Item):
    return item


if __name__ == "__main__":
    # mounting at the root path
    uvicorn.run(
        app="api:app",
        host=os.getenv("UVICORN_HOST"),  
        port=int(os.getenv("UVICORN_PORT"))
    )