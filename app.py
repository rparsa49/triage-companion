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
    "case_alex_chen": {
        "name": "Alex Chen",
        "age": 45,
        "sex": "Male",  # ⬅ NEW
        "esi_level": 1,
        "chief_complaint": "Severe, sudden-onset headache (worst ever) and neck stiffness.",
        "initial_line": "Hello, I'm Alex Chen. I have the worst headache of my life, it started about an hour ago.",
        "hot_clues": "'worst headache of life', 'sudden onset', 'neck stiffness', 'vomiting', 'photophobia'.",
        "scoring_rule": "Award +30 points for asking about any Hot Clue. Deduct -15 points for asking three or more irrelevant questions."
    },

    "case_jenny_smith": {
        "name": "Jenny Smith",
        "age": 22,
        "sex": "Female",  # ⬅ NEW
        "esi_level": 3,
        "chief_complaint": "Persistent cough and fever for three days, mild shortness of breath.",
        "initial_line": "Hi, I've had this bad cough and a fever for a few days now. I feel tired and a bit short of breath.",
        "hot_clues": "'duration of symptoms', 'recent travel or large crowds', 'O2 saturation check', 'history of asthma/smoking'.",
        "scoring_rule": "Award +25 points for asking about any Hot Clue. Deduct -5 points for asking about pain level repeatedly."
    },

    "case_roya_parsa": {
        "name": "Roya Parsa",
        "age": 68,
        "sex": "Female",  # ⬅ NEW
        "esi_level": 2,
        "chief_complaint": "Sudden onset of left-sided weakness and slurred speech.",
        "initial_line": "My name is Roya Parsa. I can barely talk and I can't feel my left arm. It started about 30 minutes ago.",
        "hot_clues": "'Time of onset (Last Known Well)', 'speech quality', 'facial droop', 'arm/leg strength comparison'.",
        "scoring_rule": "Award +40 points for establishing the time of onset or asking a key stroke symptom. Deduct -20 points for wasting time on routine history."
    }
}



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

if __name__ == '__main__':
    print(f"Starting Triage Companion Backend on [http://127.0.0.1:5000](http://127.0.0.1:5000)")
    app.run(debug=True)
