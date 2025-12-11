import base64
import os
import json
import time
from openai import OpenAI
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse
import uvicorn

app = FastAPI()

os.makedirs("uploads", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Load the OpenAI API key from environment variables
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable is not set")

# Create the client ONCE (global)
client = OpenAI(api_key=OPENAI_API_KEY)



app = FastAPI()




@app.get("/")
async def serve_index():
    # Serve the index.html file from the same folder
    return FileResponse("index.html")


PROMPT_TEXT = """
You are an expert home systems and DIY repair technician.

You help homeowners diagnose and fix problems with:
- water softeners
- lawn and landscaping (grass, plants, sprinklers, irrigation)
- dishwashers
- refrigerators
- pool and spa equipment
- furnaces and HVAC
- general home issues (leaks, noises, odd behavior)

You will be given:
- A photo of the problem area or device
- A short description of the issue
- A category label such as "water_softener", "lawn", "dishwasher", "refrigerator", "pool", "hvac", or "general"

Your job:
1. Use the category + photo + description to understand what you're looking at.
2. Identify what the device/area is (brand/model if possible, or at least type).
3. List the top 1–3 likely issues causing the problem in plain language.
4. Give clear step-by-step DIY instructions the homeowner can safely try.
5. Call out any safety warnings or things they should NOT do.
6. Tell them when it's time to call a professional instead.

Output your answer in this JSON format ONLY:

{
  "device_summary": "string",
  "likely_issues": [
    "string"
  ],
  "diy_steps": [
    "string"
  ],
  "warnings": [
    "string"
  ],
  "when_to_call_pro": "string"
}

Assume the homeowner is not technical. Be simple, calm, and specific.
"""


@app.post("/api/diagnose")
async def diagnose(
    photo: UploadFile = File(...),
    description: str = Form(...),
    category: str = Form("general")
):

    image_bytes = await photo.read()
    # Save the uploaded image to disk with a timestamped filename
    timestamp = int(time.time())
    safe_name = "".join(
        c for c in (photo.filename or "upload.jpg")
        if c.isalnum() or c in ("-", "_", ".")
    ) or "upload.jpg"

    image_path = os.path.join("uploads", f"{timestamp}_{safe_name}")
    with open(image_path, "wb") as f:
        f.write(image_bytes)


    # Convert image to base64 so we can send it as inline data
    b64_image = base64.b64encode(image_bytes).decode("utf-8")

    try:
        # Call the OpenAI multimodal model
        response = client.chat.completions.create(
            model="gpt-4.1-mini",  # or another available multimodal model
            messages=[
                {
                    "role": "system",
                    "content": PROMPT_TEXT
                },
               {
    "role": "user",
    "content": [
        {
            "type": "text",
            "text": f"Category: {category}\nDescription: {description or 'help me with this'}"
        },
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{b64_image}"
            }
        }
    ]
}

            ],
            temperature=0.2,
        )

        raw_content = response.choices[0].message.content

        # Sometimes the model might wrap JSON in ```json ... ```; strip that if needed
        cleaned = raw_content.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            # remove leading 'json' if present
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()

        data = json.loads(cleaned)

        # Basic sanity defaults, in case something is missing
        result = {
            "device_summary": data.get("device_summary", "No summary available"),
            "likely_issues": data.get("likely_issues", []),
            "diy_steps": data.get("diy_steps", []),
            "warnings": data.get("warnings", []),
            "when_to_call_pro": data.get("when_to_call_pro", "")
        }

        # Log this case to a JSONL log file
        log_entry = {
            "timestamp": timestamp,
            "category": category,
            "description": description,
            "image_path": image_path,
            "raw_model_content": raw_content,
            "parsed": result,
        }
        with open(os.path.join("logs", "cases.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")



        return JSONResponse(result)

    except Exception as e:
        # If something goes wrong, log it and return a friendly message
        print("Error in /api/diagnose:", e)
        fallback = {
            "device_summary": "We had trouble analyzing this image.",
            "likely_issues": [
                "The AI service had an error, or the photo wasn't clear enough to interpret."
            ],
            "diy_steps": [
                "Try taking a clearer photo showing the entire softener and control head.",
                "Make sure any display, labels, or error codes are visible.",
                "Resubmit the photo and description."
            ],
            "warnings": [
                "Do not attempt major disassembly if you're unsure what you're doing."
            ],
            "when_to_call_pro": "If you continue to have issues or leaks, contact a local water treatment professional."
        }
        return JSONResponse(fallback, status_code=200)



if __name__ == "__main__":
    # This allows you to run: python main.py
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)