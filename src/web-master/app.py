import json
from typing import List
import time

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
  print("Requested to train: ", name)
  response = requests.post(f"{config['ml_service_url']}/train/{name}")
  return response

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
            gr.Textbox(person_name, label="Name")
            gr.Number(get_image_count(person_name), label="Picture count")
            gr.Checkbox(person_name in trained, interactive=False, label="Trained")
            train_b = gr.Button("Train model")
            train_b.click(fn=lambda: post_train_person(person_name))

  return demo

demo = create_app()

if __name__ == "__main__":
  demo.launch(server_name="0.0.0.0")
