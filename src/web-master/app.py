import json
from typing import List
import time

import gradio as gr
import requests
from PIL import Image

with open("config.json", "r") as config_file:
  config = json.load(config_file)


def get_person_names() -> List[str]:
  response = requests.get(f"{config['storage_service_url']}/persons")
  return response.json()["persons"]

def get_image_count(person_name: str) -> int:
  response = requests.get(
    f"{config['storage_service_url']}/image-count",
    params={"person_name": person_name},
  )
  return response.json()["image_count"]

def post_person(name: str, images: List[bytes]) -> None:
  params = {
    "person_name": name,
    "images": images
  }
  requests.post(f"{config['storage_service_url']}/person", params=params)

def persons_managing_interface():
  for person_name in get_person_names():
    with gr.Row():
      gr.Textbox(person_name, label="Name")
      gr.Number(get_image_count(person_name), label="Picture count")

def create_app():
  demo = gr.Blocks()
  demo.max_file_size = 20 * 1024 * 1024
  
  with demo:
    with gr.Tab("Add a person"):
      gr.Markdown("Upload as many pictures of a person as you can")
      images = gr.File(file_count="multiple", file_types=["image"], type="binary", label="Images")
      person_name = gr.Textbox(label="Person name")
      b = gr.Button("Add")
    with gr.Tab("Manage added persons"):
      persons_managing_interface()

    b.click(
      fn=post_person,
      inputs=[person_name, images],
    )
  return demo
  
demo = create_app()
app = demo.app

if __name__ == "__main__":
  demo.launch()