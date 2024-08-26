from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

app = Flask(__name__)

# Load the model and tokenizer
model_path = "./model/h2ogpt-oasst1-falcon-40b"
if not os.path.exists(model_path):
    os.makedirs(model_path, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained("h2oai/h2ogpt-oasst1-falcon-40b", cache_dir=model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("h2oai/h2ogpt-oasst1-falcon-40b", cache_dir=model_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
else:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)

@app.route('/ask', methods=['POST'])
def ask_question():
    try:
        data = request.get_json()
        question = data['question']
        
        # Tokenize the input question
        inputs = tokenizer(question, return_tensors="pt").to("cuda")
        
        # Generate the response
        outputs = model.generate(inputs["input_ids"], max_new_tokens=100)
        
        # Decode the generated tokens
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return jsonify({"question": question, "answer": answer})
    except torch.cuda.OutOfMemoryError:
        return jsonify({"error": "CUDA Out of Memory. Try reducing the input size or batch size."}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8000))
    app.run(host='0.0.0.0', port=port)


