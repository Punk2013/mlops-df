import json
from typing import List
import time
import io

import gradio as gr
import requests
from PIL import Image

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

with open("config.json", "r") as config_file:
  config = json.load(config_file)

def get_trained_persons():
  response = requests.get(f"{config['ml_service_url']}/trained-list")
  return response.json().get("names")

def get_person_names() -> List[str]:
  response = requests.get(f"{config['storage_service_url']}/persons")
  return response.json()["persons"]

def get_image_count(person_name: str) -> int:
  response = requests.get(
    f"{config['storage_service_url']}/image-count",
    params={"person_name": person_name},
  )
  return response.json()["image_count"]

def post_person(name: str, images: List[str]):
  params = {
    "person_name": name,
  }
  files = []
  for image in images:
    files.append(("files", open(image, "rb")))
  requests.post(f"{config['storage_service_url']}/person", params=params, files=files)

def post_train_person(name):
  requests.post(f"{config['ml_service_url']}/train/{name}")
  
def get_swap_face(image, name):
  file = {"image": open(image, "rb")}
  response = requests.get(f"{config['ml_service_url']}/inference/{name}", files=file)
  img_byte_arr = io.BytesIO(response.content)
  pil_image = Image.open(img_byte_arr)
  return pil_image
  
def delete_person(name):
  if name in get_trained_persons():
    requests.delete(f"{config['ml_service_url']}/model/{name}")
  requests.delete(f"{config['storage_service_url']}/person/{name}")

def create_app():
  demo = gr.Blocks()
  demo.max_file_size = 20 * 1024 * 1024

  with demo:
    with gr.Tab("Add a person"):
      gr.Markdown("Upload as many pictures of a person as you can")
      images = gr.File(file_count="multiple", file_types=["image"], label="Images")
      person_name = gr.Textbox(label="Person name")
      b = gr.Button("Add")
      b.click(
        fn=post_person,
        inputs=[person_name, images]
      )
    with gr.Tab("Manage added persons"):
      @gr.render()
      def persons_managing_interface():
        trained = get_trained_persons()
        print(trained)
        for person_name in get_person_names():
          with gr.Row():
            name_textbox = gr.Textbox(person_name, label="Name")
            gr.Number(get_image_count(person_name), label="Picture count")
            gr.Checkbox(person_name in trained, interactive=False, label="Trained")
            gr.Button("Train model").click(fn=post_train_person, inputs=name_textbox)
            gr.Button("Remove").click(fn=delete_person, inputs=name_textbox)
    with gr.Tab("Swap face"):
      @gr.render()
      def face_swaping_interface():
        image = gr.File(file_types=["image"], label="Image of the person whose face to swap")
        person = gr.Radio(choices=get_trained_persons(), label="Choose the person whose face to use")
        swap_button = gr.Button("Swap")
        gen_img = gr.Image()
        swap_button.click(fn=get_swap_face, inputs=[image, person], outputs=gen_img)
  return demo

demo = create_app()

if __name__ == "__main__":
  demo.launch(server_name="0.0.0.0")
