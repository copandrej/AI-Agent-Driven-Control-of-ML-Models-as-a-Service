import numpy as np
from fastapi import FastAPI, Request

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Service is up!"}

@app.post("/echo")
async def echo(request: Request):
    size = 1000
    try:
        body = await request.json()
    except Exception:
        body = await request.body()
        body = body.decode()
    
    # Simulate computational load of a neural network inference
    matrix_a = np.random.rand(size, size)
    matrix_b = np.random.rand(size, size)
    # Perform matrix multiplication
    for i in range(10):
        _ = matrix_a @ matrix_b

    # echo back the received payload
    return {"echo": body}
