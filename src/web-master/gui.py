import gradio as gr
from PIL import Image

from ..storage.storage import PersonDataManager

data = PersonDataManager("./data")


def persons_managing_interface():
    for person_name in data.list_persons():
        with gr.Row():
            gr.Textbox(person_name, label="Name")
            gr.Number(data.get_image_count(person_name), label="Picture count")


with gr.Blocks() as app:
    with gr.Tab("Add a person"):
        gr.Markdown("Upload as many pictures of a person as you can")
        images = gr.File(file_count="multiple", file_types=["image"], label="Images")
        person_name = gr.Textbox(label="Person name")
        b = gr.Button("Add")
    with gr.Tab("Manage added persons"):
        persons_managing_interface()

    b.click(
        fn=lambda name, files: data.create_person_folder(
            name, [Image.open(f) for f in files]
        ),
        inputs=[person_name, images],
    ).then(persons_managing_interface)


if __name__ == "__main__":
    app.launch()
