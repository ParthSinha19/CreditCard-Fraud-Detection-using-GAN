# %%
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import matplotlib as plt
import os
# %% 
app = FastAPI()
# %%
Model_path = r"C:\Users\HP\Desktop\My Documents\acharya shit - Copy\Major Project\GAN CreditCard\generator.keras"
generator = tf.keras.models.load_model(Model_path)
print("Debug: Model Loaded Successfully!")
# %%
class TransactionRequest(BaseModel):
    card_number : str
# %%
@app.get("/")
def read_root():
    return("message:Model is running")
# %%
@app.post("/generate")
def generate_transaction(data:TransactionRequest):
    try:
        seed = len(data.card_number)
        np.random.seed(seed)

        input_dim = generator.input_shape[1]
        noise = np.random.normal(0,1(1, input_dim))
        synthetic_data = generator.predict(noise)[0]

        confidence_scores = np.clip(synthetic_data, 0, 1)


        plt.figure(figsize=(10,5))
        plt.plot(confidence_scores, marker ='o', label = "Confidence Scores")
        plt.title(f"Confidence Thresholding for Card Number: {data.card_number}")
        plt.xlabel("Feature Index")
        plt.ylabel("Confidence")
        plt.axhline(0.5, color= 'red', linestyle = '--', label= "Threshold (0.5)")
        plt.legend()
        plot_path = "Confidence.graph.png"
        plt.save(plot_path)
        plt.close()

        return FileResponse(plot_path, media_type= "image/png", filename= "confidence_graph.png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))