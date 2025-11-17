import os
import re
import json
import uuid
from datetime import datetime
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template
from google import genai
from google.genai import types


# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

app = Flask(__name__)

MODEL_NAME = 'gemini-2.5-flash'
client = genai.Client(api_key=GEMINI_API_KEY)

session_data = {}

BASE_PROMPT_TEMPLATE = """
You are a high-fidelity patient simulation for a medical triage training system.
Your name is {name} and your profile is an **ESI Level {esi_level}** case.
Your chief complaint is: {chief_complaint}.

**Triage Instructions:**
1. **Be Conversational:** Respond naturally, reflecting your age, anxiety, and symptoms.
2. **Be Truthful:** Only provide information the student asks for, based on your profile.
3. **Do NOT** volunteer the diagnosis, ESI level, or scoring information.
4. **Hot Clues (Student earns points):** {hot_clues}.
5. **Cold Clues (Student loses points/wastes time):** Asking irrelevant questions or ordering unnecessary tests.

**Scoring Rule for the AI:** {scoring_rule}

**CRITICAL OUTPUT RULE:** After your full conversational text response, you MUST append a hidden JSON object enclosed in the <SCORING_DATA> tags.
Example: [Patient's conversational text] <SCORING_DATA> {{"score_update": 20, "hot_clue_status": "Found key symptom: Sudden onset."}} </SCORING_DATA>

Begin the simulation. Introduce yourself and state your initial complaint.
"""

PATIENT_CASES = {
    "case_pregnant_abdominal_pain": {
        "name": "Angie Smith",
        "age": 26,
        "sex": "Female",
        "esi_level": 2,
        "chief_complaint": "3-day history of lower abdominal pain during 18-week pregnancy; new nausea, vomiting, fever, and urinary symptoms.",
        "initial_line": "I’ve had lower belly pain for three days, and since last night I've been nauseous and vomiting. I feel feverish and weak. I'm 18 weeks pregnant.",

        "hot_clues": (
            "'fever', 'dysuria or urinary frequency', 'incomplete bladder emptying', "
            "'nausea/vomiting', 'CVA tenderness', 'pregnancy status', 'hydration status'."
        ),

        "scoring_rule": (
            "Award +40 points for asking about fever pattern, urinary symptoms, vomiting severity, "
            "or flank/CVA tenderness. "
            "Award +30 points for confirming pregnancy complications (vaginal bleeding, fetal movement, OB history). "
            "Deduct -20 points for ignoring red flags like high fever, tachycardia, dehydration, or flank pain."
        ),
    },
    "case_leg_erythema_pain": {
        "name": "Tyler James",
        "age": 62,
        "sex": "Male",
        "esi_level": 2,
        "chief_complaint": "Progressively worsening pain and redness of the right lower leg for 2 days.",
        "initial_line": (
            "My right leg started hurting two days ago and it's gotten worse. "
            "A few hours ago I started to feel sick all over—aches, chills. "
            "The red area on my leg is getting bigger."
        ),

        "hot_clues": (
            "'fever', 'rapid progression of redness', 'erythema borders', 'lymphangitic streaking', "
            "'leg swelling asymmetry', 'recent saphenous vein harvest', 'diabetes history', "
            "'pain out of proportion', 'crepitus', 'bullae'."
        ),

        "scoring_rule": (
            "Award +40 points for asking about rate of progression, fever/chills, prior skin breaks, "
            "or history of saphenous vein harvest. "
            "Award +30 points for distinguishing cellulitis versus DVT (calf asymmetry, tenderness, risk factors). "
            "Deduct -20 points for failing to evaluate for necrotizing soft tissue infection (pain severity, crepitus). "
            "Deduct -15 points for ignoring cardiovascular or diabetes-related complications."
        ),
    },

    "case_adolescent_back_pain": {
        "name": "Alex Paul",
        "age": 12,
        "sex": "Male",
        "esi_level": 3,
        "chief_complaint": "Occasional mild upper-back pain and stiffness for ~1 year.",
        "initial_line": "Sometimes my upper back hurts and feels stiff, especially in the morning, but mostly I'm fine.",
        "hot_clues": (
            "'pain onset during growth', 'shoulder asymmetry', 'forward-bend rib hump', "
            "'family history of scoliosis', 'neurologic symptoms (weakness, gait change)', 'night pain', "
            "'rapid progression of curve'."
        ),
        "scoring_rule": (
            "Award +30 points for asking about physical exam signs (scoliosis screening, Adam’s forward-bend test), "
            "or family history, or neurologic symptoms. "
            "Award +20 points for ruling out red-flag symptoms (night pain, weight loss, bowel/bladder changes). "
            "Deduct –15 points for focusing only on pain severity and ignoring structural assessment."
        )
    }
    
}

def remove_bracketed_text(text):
    # Removes anything inside (), [], {}, <> including nested cases
    return re.sub(r'\s*[\(\[\{<][^)\]\}>]*[\)\]\}>]\s*', ' ', text).strip()

# Parse Gemini Response
def parse_gemini_response(raw_text, current_score):
    """
    Extracts the clean patient text and the scoring JSON from the raw Gemini output.
    Uses regex to find the hidden <SCORING_DATA> block.
    """
    # Regex to find the JSON block enclosed in the custom tags
    score_match = re.search(r'<SCORING_DATA>\s*({.*?})\s*</SCORING_DATA>', raw_text, re.DOTALL)

    patient_message = raw_text.split('<SCORING_DATA>')[0].strip()
    new_score = current_score
    clue_status = "Awaiting key finding..."

    if score_match:
        try:
            # Safely load the JSON string from the regex group
            scoring_data = json.loads(score_match.group(1))

            # Update values based on the extracted JSON
            new_score = scoring_data.get('score_update', current_score)
            clue_status = scoring_data.get(
                'hot_clue_status', 'Feedback received.')

        except json.JSONDecodeError:
            clue_status = "Error processing AI feedback (JSON format broken)."

    return patient_message, new_score, clue_status


# Function to initialize the patient simulation
def start_new_triage_session(case_id):
    """Initializes a new Gemini chat session for a specific patient case."""

    case_data = PATIENT_CASES.get(case_id)
    if not case_data:
        raise ValueError(f"Case ID '{case_id}' not found.")

    full_prompt = BASE_PROMPT_TEMPLATE.format(
        name=case_data["name"],
        esi_level=case_data["esi_level"],
        chief_complaint=case_data["chief_complaint"],
        hot_clues=case_data["hot_clues"],
        scoring_rule=case_data["scoring_rule"]
    )

    chat = client.chats.create(model=MODEL_NAME)
    chat.send_message(full_prompt)

    session_id = str(uuid.uuid4())

    # Arrival time = when the triage session starts
    arrival_time = datetime.now().strftime("%I:%M:%S %p")

    session_data[session_id] = {
        'chat': chat,
        'case_id': case_id,
        'current_score': 0,
        'patient_name': case_data["name"],
        'esi_goal': case_data["esi_level"],
        'arrival_time': arrival_time
    }

    return session_id, case_data["initial_line"], case_data, arrival_time

# --- API Endpoints ---
@app.route('/')
def index():
    """Renders the main triage interface page and passes case list."""

    cases_for_frontend = []

    for k, v in PATIENT_CASES.items():
        full_complaint = v['chief_complaint']
        MAX_LEN = 30

        if len(full_complaint) > MAX_LEN:
            display_complaint = full_complaint[:MAX_LEN] + '...'
        else:
            display_complaint = full_complaint

        cases_for_frontend.append({
            'id': k,
            'name': v['name'],
            'complaint': display_complaint 
        })

    return render_template('index.html', cases=cases_for_frontend)

@app.route('/start', methods=['POST'])
def start_session():
    """Endpoint to start a new simulation session for a specified case."""
    data = request.json
    case_id = data.get('case_id')

    if not case_id:
        return jsonify({'error': 'No case ID provided'}), 400

    try:
        session_id, initial_patient_text, case_data, arrival_time = start_new_triage_session(case_id)

        return jsonify({
            'session_id': session_id,
            'patient_text': initial_patient_text,
            'case_id': case_id,
            'patient_name': case_data["name"],
            'esi_goal': case_data["esi_level"],
            'initial_score': 0,
            'age': case_data["age"],
            'sex': case_data["sex"],
            'arrival_time': arrival_time,
            'chief_complaint': case_data["chief_complaint"]
        })
    except ValueError as e:
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        return jsonify({'error': f"Server error starting session: {str(e)}"}), 500

@app.route('/triage', methods=['POST'])
def triage_turn():
    """Endpoint for a user's turn in the conversation, handling AI response and scoring."""
    data = request.json
    session_id = data.get('session_id')
    user_message = data.get('message')

    if session_id not in session_data:
        return jsonify({'error': 'Invalid or expired session ID'}), 404

    session = session_data[session_id]
    chat = session['chat']

    try:
        # Send the student's message to the Gemini API
        gemini_response = chat.send_message(user_message)
        raw_text = gemini_response.text

        # 1. Parse the AI's response to separate patient text and structured feedback
        patient_message, new_score, clue_status = parse_gemini_response(
            raw_text,
            session['current_score']
        )

        # 2. Update and save the score in the session data
        session['current_score'] = new_score

        # 3. Return the patient response and the real-time feedback
        return jsonify({
            'patient_text': patient_message,
            'current_score': new_score,
            'real_time_feedback': clue_status
        })

    except Exception as e:
        # Log the error for debugging
        print(f"Error during triage turn for session {session_id}: {e}")
        return jsonify({'error': 'An internal error occurred during the conversation turn.'}), 500

@app.route('/transcribe_audio', methods=['POST'])
def transcribe_audio():
    data = request.form  # Using form + files
    session_id = data.get('session_id')
    if session_id not in session_data:
        return jsonify({'error': 'Invalid or expired session ID'}), 404

    # Expect the audio file in request.files
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file uploaded'}), 400

    audio_file = request.files['audio']
    audio_bytes = audio_file.read()
    mime_type = audio_file.mimetype or 'audio/wav'

    try:
        # You can upload using Files API if size >20MB
        uploaded = client.files.upload(file=audio_bytes, mime_type=mime_type)

        # Build prompt: ask to transcribe
        prompt = "Please transcribe this audio recording of a triage question."

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[prompt, uploaded]
        )

        transcript = response.text.strip()
        return jsonify({'transcript': transcript})

    except Exception as e:
        return jsonify({'error': f"Error during transcription: {str(e)}"}), 500
    
@app.route('/synthesize_speech', methods=['POST'])
def synthesize_speech():
    """Converts patient text into speech using Gemini TTS and returns base64 WAV audio."""
    data = request.json or {}
    text_raw = (data.get("text") or "").strip()
    text = remove_bracketed_text(text_raw)


    if not text:
        return jsonify({'error': 'No text provided'}), 400

    try:
        # Call Gemini TTS model (PCM16 output)
        response = client.models.generate_content(
            model="gemini-2.5-flash-preview-tts",
            contents=text,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name="Kore"
                        )
                    )
                ),
            ),
        )

        # Get the inline audio data (PCM16)
        part = response.candidates[0].content.parts[0]
        inline = getattr(part, "inline_data", None)

        if not inline or inline.data is None:
            print("TTS: No inline_data in response:", response)
            return jsonify({"error": "No audio data in TTS response"}), 500

        pcm_bytes = inline.data  # raw PCM16 from Gemini
        mime_in = inline.mime_type or "audio/L16;codec=pcm;rate=24000"

        # Wrap PCM into a WAV container so the browser can play it
        import io
        import wave
        import base64

        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wf:
            wf.setnchannels(1)      # mono
            wf.setsampwidth(2)      # 16-bit
            wf.setframerate(24000)  # 24 kHz
            wf.writeframes(pcm_bytes)

        wav_bytes = buffer.getvalue()
        audio_b64 = base64.b64encode(wav_bytes).decode("utf-8")

        return jsonify({
            "audio_base64": audio_b64,
            "mime_type": "audio/wav"
        })

    except Exception as e:
        print("TTS error:", e)
        return jsonify({"error": f"TTS synthesis failed: {str(e)}"}), 500


if __name__ == '__main__':
    print(f"Starting Triage Companion Backend on [http://127.0.0.1:5000](http://127.0.0.1:5000)")
    app.run(debug=True)
