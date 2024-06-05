from flask import Flask, render_template, request
from llama_cpp import Llama
from huggingface_hub import hf_hub_download

# Flask app initialization
app = Flask(__name__)

# Model names and files
model_info = [
    {
        "model_name": "Fawazzx/SaulLM_semantic_finetuned-7b-v0.1.gguf",
        "model_file": "SaulLM_semantic_finetuned-7b-v0.1.gguf"
    },
    {
        "model_name": "Fawazzx/Finetuned-saulLM-7b-v1.5.gguf",
        "model_file": "Finetuned-saulLM-7b-v1.5.gguf"
    }
]

# Initialize the models with error handling
llm_models = []
for info in model_info:
    try:
        model_path = hf_hub_download(info["model_name"], filename=info["model_file"])
        llm = Llama(model_path=model_path, n_gpu_layers=-1)
        llm_models.append(llm)
    except FileNotFoundError:
        print(f"Error: Model file not found for {info['model_name']}. Please check the path.")
        exit(1)
    except Exception as e:
        print(f"Error initializing LLM {info['model_name']}: {e}")
        exit(1)

# Assign models to variables for easier access
llm_1, llm_2 = llm_models

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict_role", methods=["POST"])
def predict_role():
    sentence = request.form.get("sentence")
    if not sentence:
        return "Please enter a sentence."

    # Prepare prompt for model 1
    prompt = f"""[LEGAL] Analyze the sentence: {sentence}.

**Task:** Identify the semantic role in the Indian legal context for the sentence above.

Possible roles: Arguments, Precedent, Statutes, Facts, Ratio Decidendi, Ruling of Lower Court, Ruling of Present Court
[/LEGAL]"""

    # Use the first model to obtain the predicted role
    try:
        output = llm_1(prompt=prompt, max_tokens=300)
        predicted_role = output['choices'][0]['text'].strip()
        # Format the role with HTML tags for bold
        predicted_role = predicted_role.replace("**", "<strong>").replace("**", "</strong>")
    except Exception as e:
        print(f"Error during prediction with model 1: {e}")
        return "An error occurred during prediction. Please try again later."

    return render_template("index.html", sentence=sentence, prediction_role=predicted_role)

@app.route("/predict_section", methods=["POST"])
def predict_section():
    case_details = request.form.get("case_details")
    if not case_details:
        return "Please enter case details."

    # Prepare prompt for model 2
    prompt = f"Given a description of a legal case in India, what is the most relevant legal section it falls under? Here's the case: {case_details}"

    try:
        output = llm_2(prompt=prompt, max_tokens=300)
        full_text = output['choices'][0]['text'].strip()

        # Extract the text after [/INST]
        if '[/INST]' in full_text:
            predicted_section = full_text.split('[/INST]')[1].strip()
        else:
            predicted_section = full_text  # In case [/INST] is not found, use the full text

    except Exception as e:
        print(f"Error during prediction with model 2: {e}")
        return "An error occurred during prediction. Please try again later."

    return render_template("index.html", case_details=case_details, prediction_section=predicted_section)

if __name__ == "__main__":
    app.run(debug=True)
