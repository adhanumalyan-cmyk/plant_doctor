
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():

from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():

    return {"message": "Hello, Plant Doctor!"}