import time

import dearpygui.dearpygui as dpg
import torch
import transformers
from PIL import Image

MODEL_DIR = "./models/resnet-pcb-ext-70-30/"
IMAGE_PROCESSOR_CLASS = transformers.AutoImageProcessor
IMAGE_CLASSIFICATION_CLASS = transformers.ResNetForImageClassification
LABELS = ["defective", "ok"]
TEXTURE_TAG = "texture"
TEXTURE_SIZE = 512
IMAGE_SIZE = 256
CLASSIFICATION_TAG = "classification"
CONFIDENCE_TAG = "confidence"
RUNTIME_TAG = "runtime"
PROCESSOR = None
MODEL = None


def main() -> None:
    setup_model()
    dpg.create_context()
    dpg.create_viewport(title="Quality Control", width=600, height=600)
    dpg.set_global_font_scale(1.5)

    gui()

    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("Primary", True)
    dpg.start_dearpygui()
    dpg.destroy_context()


def gui():
    setup_file_dialog()
    setup_image()

    with dpg.window(tag="Primary"):
        dpg.add_button(
            label="Open Image", callback=lambda: dpg.show_item("file_dialog_id")
        )
        dpg.add_image(TEXTURE_TAG, width=IMAGE_SIZE, height=IMAGE_SIZE)
        dpg.add_text("Classification:", tag=CLASSIFICATION_TAG)
        dpg.add_text("Confidence:", tag=CONFIDENCE_TAG)
        dpg.add_text("Runtime:", tag=RUNTIME_TAG)


def setup_file_dialog():
    with dpg.file_dialog(
        directory_selector=False,
        file_count=1,
        show=False,
        callback=open_file_callback,
        id="file_dialog_id",
        width=700,
        height=400,
    ):
        ...
        dpg.add_file_extension(".jpeg")
        dpg.add_file_extension(".png")
        dpg.add_file_extension(".jpg")


def setup_image():
    texture_data = []
    for i in range(0, TEXTURE_SIZE**2):
        texture_data.append(0)
        texture_data.append(0)
        texture_data.append(0)
        texture_data.append(0)

    with dpg.texture_registry():
        dpg.add_dynamic_texture(
            tag=TEXTURE_TAG,
            width=TEXTURE_SIZE,
            height=TEXTURE_SIZE,
            default_value=texture_data,
        )


def setup_model():
    global PROCESSOR
    PROCESSOR = IMAGE_PROCESSOR_CLASS.from_pretrained(MODEL_DIR)

    global MODEL
    MODEL = IMAGE_CLASSIFICATION_CLASS.from_pretrained(
        MODEL_DIR,
        num_labels=len(LABELS),
        id2label={str(i): c for i, c in enumerate(LABELS)},
        label2id={c: str(i) for i, c in enumerate(LABELS)},
        ignore_mismatched_sizes=True,
    )


def open_file_callback(_sender, appdata):
    if not appdata or not appdata["selections"]:
        return

    img_path = next(iter(appdata["selections"].values()))
    _width, _height, _channels, data = dpg.load_image(img_path)

    with dpg.texture_registry():
        dpg.set_value(TEXTURE_TAG, data)

    start = time.perf_counter()
    with Image.open(img_path) as img:
        inputs = PROCESSOR(img, return_tensors="pt")

    with torch.no_grad():
        logits = MODEL(**inputs).logits

    label_idx = logits.argmax(-1).item()
    classfication = LABELS[label_idx]
    confidence = logits.softmax(-1)[0][label_idx] * 100

    dpg.set_value(CLASSIFICATION_TAG, f"Classification: {classfication}")
    dpg.set_value(CONFIDENCE_TAG, f"Confidence: {confidence:.2f}%")
    dpg.set_value(RUNTIME_TAG, f"Runtime: {(time.perf_counter() - start)*1000:.2f}ms")


if __name__ == "__main__":
    main()
