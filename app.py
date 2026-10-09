
from pathlib import Path

import streamlit as st
import torch

from model import GPTLanguageModel
from tokenizer import CharTokenizer


CHECKPOINT_PATH = Path(__file__).parent / "checkpoints" / "model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


st.set_page_config(
    page_title="My GPT",
    page_icon="🤖",
    layout="centered",
)


@st.cache_resource
def load_model():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    # Use only a checkpoint you trust.
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    model = GPTLanguageModel(
        vocab_size=checkpoint["vocab_size"],
        block_size=checkpoint["block_size"],
        n_embd=checkpoint["n_embd"],
        n_head=checkpoint["n_head"],
        n_layer=checkpoint["n_layer"],
        dropout=checkpoint["dropout"],
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    tokenizer = CharTokenizer(
        stoi=checkpoint["stoi"],
        itos=checkpoint["itos"],
    )

    return model, tokenizer, checkpoint


st.title("🤖 My GPT")
st.caption("A character-level Transformer trained from scratch")

try:
    model, tokenizer, checkpoint = load_model()
except Exception as exc:
    st.error(f"Could not load the model: {exc}")
    st.stop()

st.success("Model loaded successfully.")

prompt = st.text_area(
    "Enter your prompt",
    value="ROMEO:",
    height=100,
)

col1, col2 = st.columns(2)

with col1:
    max_new_tokens = st.slider(
        "New characters",
        min_value=50,
        max_value=500,
        value=200,
        step=50,
    )

with col2:
    temperature = st.slider(
        "Temperature",
        min_value=0.2,
        max_value=1.5,
        value=0.8,
        step=0.1,
    )

if st.button("Generate text", type="primary", use_container_width=True):
    try:
        encoded = tokenizer.encode(prompt)

        if not encoded:
            encoded = [0]

        context = torch.tensor(
            [encoded],
            dtype=torch.long,
            device=DEVICE,
        )

        with torch.inference_mode():
            output = model.generate(
                context,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )

        generated_text = tokenizer.decode(output[0].tolist())

        st.subheader("Generated text")
        st.text_area(
            "Output",
            value=generated_text,
            height=350,
        )

        st.download_button(
            "Download generated text",
            data=generated_text,
            file_name="generated_text.txt",
            mime="text/plain",
        )

    except ValueError as exc:
        st.error(str(exc))
    except Exception as exc:
        st.error(f"Generation failed: {exc}")

with st.expander("Model information"):
    st.write(f"Device: {DEVICE}")
    st.write(f"Vocabulary size: {checkpoint['vocab_size']}")
    st.write(f"Embedding dimension: {checkpoint['n_embd']}")
    st.write(f"Transformer layers: {checkpoint['n_layer']}")
    st.write(f"Attention heads: {checkpoint['n_head']}")