import gradio as gr
from PIL import Image
import requests
import io


def recognize_digit(image):
    # Convert to PIL Image necessary if using the API method
    # image = image["composite"]
    # image = Image.fromarray(image.astype("uint8"))
    # image = image.convert("L")
    # img_binary = io.BytesIO()
    # image.save(img_binary, format="PNG")

    buf = io.BytesIO()
    image.save(buf, format="PNG")  # Convertir en format PNG
    img_bytes = buf.getvalue()

    response = requests.post(
        "http://127.0.0.1:5000/predict", data=img_bytes
    )
    # response = requests.post(
    #     "http://127.0.0.1:5000/predict", data=img_binary.getvalue()
    # )
    predict = response.json()["prediction"]
    return predict


if __name__ == "__main__":
    gr.Interface(
        fn=recognize_digit,
        inputs=gr.Image(type="pil"),
        outputs="text",
        # live=True,
        description="Download a poster image and get the predicted genre of the movie",
    ).launch(debug=True, share=True)
