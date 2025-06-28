import os
from flask import Flask, request, render_template_string
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering

app = Flask(__name__)

# Load BLIP model 
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")

with open('index.html', 'r') as file:
    HTML = file.read()

def get_answer_blip(model, processor, image, question):
    inputs = processor(image, question, return_tensors="pt")
    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)

@app.route("/", methods=["GET", "POST"])
def index():
    answer = None
    if request.method == "POST":
        file = request.files["image"]
        question = request.form["question"]
        if file and question:
            image = Image.open(file.stream).convert("RGB")
            answer = get_answer_blip(model, processor, image, question)
    return render_template_string(HTML, answer=answer)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)