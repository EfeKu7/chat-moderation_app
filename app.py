import argparse
from functools import lru_cache
from pathlib import Path
from time import perf_counter

import gradio as gr
import torch

from moderation import FILES, MODELS, Moderator


def create_app(artifacts):
    @lru_cache(maxsize=len(MODELS))
    def load_model(name):
        try:
            return Moderator(name, artifacts)
        except FileNotFoundError as exc:
            raise gr.Error("Model files are missing. Follow the training steps in the README first.") from exc

    def moderate(message, model_name, messages, timings):
        message = (message or "").strip()
        if not message:
            return "", messages, timings, messages, timings

        model = load_model(model_name)
        start = perf_counter()
        label, confidence = model.classify(message)
        latency = (perf_counter() - start) * 1000
        visible_message = "[Message masked]" if label == "OFFENSIVE" else message
        messages = messages + [
            {"role": "assistant", "content": f"{label} · {confidence:.1%}\n\n{visible_message}"}
        ]
        timings = timings + [[len(messages), model_name, label, round(latency, 2)]]
        return "", messages, timings, messages, timings

    with gr.Blocks(theme=gr.themes.Default(primary_hue="indigo", secondary_hue="blue")) as demo:
        gr.Markdown("# Real-Time Chat Moderation")
        with gr.Row():
            with gr.Column(scale=3):
                chat = gr.Chatbot(label="Chat", type="messages", height=380, render_markdown=False)
                with gr.Row():
                    message = gr.Textbox(label="Message", placeholder="Type a message in Turkish",
                                         max_lines=3, scale=4)
                    send = gr.Button("Send", variant="primary", scale=1)
            with gr.Column(scale=2):
                selected = gr.Dropdown(choices=MODELS, value=MODELS[-1], label="Active model")
                history = gr.Dataframe(
                    headers=["Message", "Model", "Label", "Latency (ms)"],
                    datatype=["number", "str", "str", "number"], value=[], type="array", interactive=False,
                )
                clear = gr.Button("Clear chat")
        messages_state = gr.State([])
        timings_state = gr.State([])
        outputs = [message, chat, history, messages_state, timings_state]
        gr.on(triggers=[message.submit, send.click], fn=moderate,
              inputs=[message, selected, messages_state, timings_state], outputs=outputs,
              api_name="moderate",
              concurrency_limit=1, concurrency_id="moderation")
        clear.click(lambda: ("", [], [], [], []), outputs=outputs, api_name="clear_chat",
                    concurrency_limit=1, concurrency_id="moderation")
    return demo, load_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    missing = [filename for filename in FILES.values()
               if not (args.artifacts / filename).is_file()]
    if missing:
        parser.error(f"Missing model files in {args.artifacts}. Run python train.py first.")
    torch.set_num_threads(args.threads)
    demo, load_model = create_app(args.artifacts)
    load_model(MODELS[-1])
    demo.launch()


if __name__ == "__main__":
    main()