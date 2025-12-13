import gradio as gr

def render_dynamic_content(choice):
    if choice == "simple":
        return gr.Column([
            gr.Textbox("Simple mode"),
            gr.Button("Submit")
        ])
    else:
        return gr.Column([
            gr.Textbox("Advanced mode 1"),
            gr.Textbox("Advanced mode 2"),
            gr.Slider(0, 100),
            gr.Button("Advanced Submit")
        ])

with gr.Blocks() as demo:
    choice = gr.Radio(["simple", "advanced"], value="simple")

    # Dynamic container
    @gr.render(inputs=choice)
    def render_content(choice):
        render_dynamic_content(choice)

demo.launch()
