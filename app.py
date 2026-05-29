from dotenv import load_dotenv
load_dotenv()

import html
import os
import tempfile

import streamlit as st
from openai import OpenAI

if not os.getenv("OPENAI_API_KEY") and os.getenv("OPEN_API_KEY"):
    os.environ["OPENAI_API_KEY"] = os.getenv("OPEN_API_KEY", "")

st.set_page_config(
    page_title="Anamika Voice Agent",
    page_icon=":material/mic:",
    layout="centered",
    initial_sidebar_state="collapsed",
)

from backend import get_ai_response

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.markdown(
    """
    <style>
    :root{
        --bg:#080c10;--panel:rgba(15,23,28,.88);--panel2:rgba(24,35,42,.92);
        --line:rgba(255,255,255,.12);--text:#eef6f7;--muted:#98a7ab;
        --teal:#22c7b8;--amber:#f4b860;--user:#133f38;--agent:#19222c;
    }
    .stApp{
        background:linear-gradient(135deg,rgba(34,199,184,.22),transparent 30%),
        linear-gradient(315deg,rgba(238,111,104,.18),transparent 28%),
        radial-gradient(circle at 50% -10%,rgba(244,184,96,.20),transparent 32rem),var(--bg);
        color:var(--text);
    }
    [data-testid="stHeader"]{background:transparent}
    .block-container{max-width:900px;padding:2rem 1rem 3rem}
    .shell{
        border:1px solid var(--line);border-radius:8px;padding:24px;
        background:linear-gradient(180deg,var(--panel),rgba(8,12,16,.78));
        box-shadow:0 26px 80px rgba(0,0,0,.38);backdrop-filter:blur(16px);
    }
    .top{display:flex;justify-content:space-between;align-items:center;gap:18px;
        padding-bottom:18px;border-bottom:1px solid var(--line)}
    .brand{display:flex;gap:14px;align-items:center;min-width:0}
    .logo{
        width:54px;height:54px;border-radius:8px;display:grid;place-items:center;
        color:#061012;font-weight:900;background:linear-gradient(135deg,var(--teal),var(--amber));
        box-shadow:0 0 34px rgba(34,199,184,.24);
    }
    h1{margin:0;color:var(--text);font-size:2.15rem;line-height:1.05;letter-spacing:0}
    .sub{margin:7px 0 0;color:var(--muted);font-size:.96rem}
    .badge{
        border:1px solid rgba(34,199,184,.34);border-radius:999px;padding:8px 12px;
        color:var(--teal);background:rgba(34,199,184,.08);font-size:.84rem;font-weight:800;
    }
    .label{
        margin:22px 0 10px;color:var(--amber);font-size:.76rem;font-weight:900;
        letter-spacing:.08em;text-transform:uppercase;
    }
    div[data-testid="stRadio"]{
        border:1px solid var(--line);border-radius:8px;padding:8px 12px;background:var(--panel2);
    }
    div[data-testid="stRadio"] label,.stTextInput label{color:var(--text)}
    .stTextInput input{
        min-height:48px;border-radius:8px;color:var(--text);
        background:rgba(255,255,255,.07);border:1px solid var(--line);
    }
    .stTextInput input:focus{border-color:var(--teal);box-shadow:0 0 0 1px rgba(34,199,184,.35)}
    .stButton>button{
        width:100%;min-height:46px;border:0;border-radius:8px;
        background:linear-gradient(135deg,var(--teal),#168f84);color:#041012;font-weight:900;
    }
    .stButton>button:hover{color:#041012;background:linear-gradient(135deg,var(--amber),var(--teal))}
    .stAudioInput{
        border:1px dashed rgba(34,199,184,.42);border-radius:8px;
        padding:16px;background:rgba(34,199,184,.08);
    }
    .empty{
        padding:18px;border-radius:8px;border:1px solid var(--line);
        background:rgba(255,255,255,.05);color:var(--muted);text-align:center;
    }
    .chat{display:flex;flex-direction:column;gap:12px}.row{display:flex}
    .row.user{justify-content:flex-end}.row.agent{justify-content:flex-start}
    .bubble{
        max-width:min(78%,620px);padding:12px 14px;border-radius:8px;line-height:1.45;
        overflow-wrap:anywhere;border:1px solid var(--line);box-shadow:0 12px 28px rgba(0,0,0,.18);
    }
    .bubble.user{background:var(--user);color:#eafff6}.bubble.agent{background:var(--agent);color:var(--text)}
    .speaker{
        display:block;margin-bottom:5px;color:var(--amber);font-size:.72rem;
        font-weight:900;letter-spacing:.08em;text-transform:uppercase;
    }
    @media(max-width:640px){
        .shell{padding:18px}.top{flex-direction:column;align-items:flex-start}
        h1{font-size:1.55rem}.bubble{max-width:92%}
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def say(ai_text: str) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
        path = f.name
    with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts", voice="nova", input=ai_text
    ) as response:
        response.stream_to_file(path)
    return path


def add_message(user_text: str, ai_text: str) -> None:
    st.session_state.history.append(("You", user_text))
    st.session_state.history.append(("AI", ai_text))


def show_history() -> None:
    st.markdown('<div class="label">Conversation</div>', unsafe_allow_html=True)
    if not st.session_state.history:
        st.markdown('<div class="empty">Start a conversation with Anamika.</div>', unsafe_allow_html=True)
        return
    st.markdown('<div class="chat">', unsafe_allow_html=True)
    for role, msg in st.session_state.history:
        side = "user" if role == "You" else "agent"
        name = "You" if role == "You" else "Anamika"
        safe = html.escape(msg).replace("\n", "<br>")
        st.markdown(
            f'<div class="row {side}"><div class="bubble {side}">'
            f'<span class="speaker">{name}</span>{safe}</div></div>',
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)


def handle_voice() -> None:
    st.markdown('<div class="label">Voice Input</div>', unsafe_allow_html=True)
    audio = st.audio_input("Speak to Anamika")
    if audio is None:
        return
    with st.spinner("Transcribing and thinking..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(audio.getbuffer())
            audio_path = f.name
        with open(audio_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1", file=audio_file, language="en"
            )
        user_text = transcript.text.strip()
        if user_text and user_text != st.session_state.last_input:
            st.session_state.last_input = user_text
            ai_text = get_ai_response(user_text)
            add_message(user_text, ai_text)
            st.markdown('<div class="label">Response Audio</div>', unsafe_allow_html=True)
            st.audio(say(ai_text))


def handle_chat() -> None:
    st.markdown('<div class="label">Chat Input</div>', unsafe_allow_html=True)
    with st.form("chat_form", clear_on_submit=True):
        user_text = st.text_input(
            "Ask Anamika",
            placeholder="Ask about Anamika, 100x, skills, experience...",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Ask Anamika")
    if not (submitted and user_text.strip()):
        return
    with st.spinner("Writing a sharp answer..."):
        clean_text = user_text.strip()
        ai_text = get_ai_response(clean_text)
        add_message(clean_text, ai_text)
        st.markdown('<div class="label">Response Audio</div>', unsafe_allow_html=True)
        st.audio(say(ai_text))


if "history" not in st.session_state:
    st.session_state.history = []
if "last_input" not in st.session_state:
    st.session_state.last_input = ""

st.markdown(
    """
    <div class="shell">
        <div class="top">
            <div class="brand">
                <div class="logo">AA</div>
                <div>
                    <h1>Anamika Voice Agent</h1>
                    <p class="sub">Voice and chat, tuned for short confident answers.</p>
                </div>
            </div>
            <div class="badge">ONLINE</div>
        </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="label">Mode</div>', unsafe_allow_html=True)
mode = st.radio("Mode", ["Voice", "Chat"], horizontal=True, label_visibility="collapsed")

handle_voice() if mode == "Voice" else handle_chat()
show_history()
st.markdown("</div>", unsafe_allow_html=True)
