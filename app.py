from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastapi.middleware.cors import CORSMiddleware
import torch
import os

# Initialize the FastAPI app
app = FastAPI()

# CORS configuration
origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://yourdomain.com",  # Add your specific domains here
    "*",  # Allow all origins, be careful with this in production
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set the model path
model_path = "./model/h2ogpt-oasst1-falcon-40b"

# Download and load the model and tokenizer if not already present
if not os.path.exists(model_path):
    os.makedirs(model_path, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained("h2oai/h2ogpt-oasst1-falcon-40b", cache_dir=model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("h2oai/h2ogpt-oasst1-falcon-40b", cache_dir=model_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
else:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)

# Define the request body model
class QuestionRequest(BaseModel):
    question: str

# Error handling middleware
@app.middleware("http")
async def add_custom_header(request: Request, call_next):
    try:
        response = await call_next(request)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Define the API endpoint
@app.post("/ask")
def ask_question(request: QuestionRequest):
    try:
        # Tokenize the input question
        inputs = tokenizer(request.question, return_tensors="pt").to("cuda")
        
        # Generate the response
        outputs = model.generate(inputs["input_ids"], max_new_tokens=100)
        
        # Decode the generated tokens
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return {"question": request.question, "answer": answer}
    except torch.cuda.OutOfMemoryError:
        raise HTTPException(status_code=500, detail="CUDA Out of Memory. Try reducing the input size or batch size.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

# Run the API
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
