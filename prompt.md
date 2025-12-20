Update this method right now it uses test based llm use multimodal by passing the pdf to llm 

def call_llm_api(prompt: str, file_path: str) -> Dict[str, Any]:
    """
    Calls Gemini API with automatic retry on Rate Limit (429) errors.
    Returns parsed JSON dictionary.
    """
    _setup_environment()
    model = genai.GenerativeModel(MODEL_NAME)

    max_retries = 3
    base_wait = 30  # Start waiting 30 seconds

    for attempt in range(max_retries):
        try:
            # --- Call Gemini model ---
            response = model.generate_content(prompt)
            
            # --- Clean & parse JSON ---
            # Return immediately if successful
            return parse_llm_json(response.text)

        except ResourceExhausted as e:
            # This catches the 429 Quota Exceeded error
            wait_time = base_wait * (attempt + 1)
            print(f"\n[WARN] Gemini Quota Exceeded. Waiting {wait_time}s before retry ({attempt + 1}/{max_retries})...")
            time.sleep(wait_time)
        
        except Exception as e:
            # Other errors (auth, network) should crash immediately
            print(f"[ERROR] Gemini API Failed: {e}")
            raise e

    # If we run out of retries
    raise RuntimeError("Gemini API Quota Exceeded after multiple retries. Please check your billing/limits.")

this is how ti its done, but it is for a diffrent part
make sure path is correct
   file_path = (
        existing_json.get("llm_invoices", {}).get("file_path")
        if existing_json.get("llm_invoices")
        else None
    )
    if not file_path or not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found or invalid: {file_path}")

    # --- Load file for Gemini ---
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type is None:
        mime_type = "image/jpeg"  # fallback default
    file_obj = genai.upload_file(file_path, mime_type=mime_type)


and make code that passes the correct type like pdf jpg whatever

this is how you use  gemini crrectly
  model = genai.GenerativeModel(MODEL_NAME)
    response = model.generate_content([prompt, file_obj])
    raw_text = response.text.strip() if response and response.text else ""

    # --- Clean & parse JSON ---
    cleaned = re.sub(r"^```(?:json)?|```$", "", raw_text.strip(), flags=re.MULTILINE).strip()

    try:
        corrected_json = json.loads(cleaned)
        return corrected_json
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Gemini did not return valid JSON. Error: {e}\nRaw output:\n{raw_text}"
