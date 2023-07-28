import io
from flask import Flask, render_template, request
import cv2
import numpy as np
from PIL import Image
import base64
import urllib.request

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/watermark', methods=['POST'])
def watermark():
    # Get watermark type choice (image or text)
    watermark_type = request.form['watermark_type']

    if watermark_type == 'image':
        # Get image URL and load the image
        image_url = request.form['image_url']
        headers = {'User-Agent': 'Mozilla/5.0'}
        req = urllib.request.Request(url=image_url, headers=headers)
        image_data = urllib.request.urlopen(req).read()
        with open('image.jpg', 'wb') as f:
            f.write(image_data)
        image = Image.open('image.jpg')

        # Get watermark URL and load the logo
        watermark_url = request.form['watermark_url']
        req = urllib.request.Request(url=watermark_url, headers=headers)
        watermark_data = urllib.request.urlopen(req).read()
        with open('logo.png', 'wb') as f:
            f.write(watermark_data)
        logo = Image.open('logo.png')

        # Resize images
        image_logow = image.resize((500, 300))
        logo = logo.resize((100, 100))

        # Convert images to numpy arrays
        image_logow = np.array(image_logow.convert('RGB'))
        logo = np.array(logo.convert('RGB'))

        # Perform image watermarking
        h_image, w_image, _ = image_logow.shape
        h_logo, w_logo, _ = logo.shape

        center_y = int(h_image / 2)
        center_x = int(w_image / 2)

        top_y = center_y - int(h_logo / 2)
        left_x = center_x - int(w_logo / 2)
        bottom_y = top_y + h_logo
        right_x = left_x + w_logo

        # Get ROI
        roi = image_logow[top_y: bottom_y, left_x: right_x]

        # Reduce the intensity of the image watermark (adjust the alpha value)
        alpha = 0.8  # Adjust the alpha value (0.0 to 1.0)
        watermark = cv2.addWeighted(roi, alpha, logo, 1 - alpha, 0)

        # Replace the ROI on the image
        image_logow[top_y: bottom_y, left_x: right_x] = watermark

        # Convert the watermarked image back to PIL format
        watermarked_image = Image.fromarray(image_logow, 'RGB')

        # Convert the watermarked image to base64
        image_data = io.BytesIO()
        watermarked_image.save(image_data, format='JPEG')
        image_data.seek(0)
        encoded_image = base64.b64encode(image_data.getvalue()).decode('utf-8')

        return render_template('result.html', image=encoded_image)
    elif watermark_type == 'text':
        # Get image URL and load the image
        image_url = request.form['image_url']
        headers = {'User-Agent': 'Mozilla/5.0'}
        req = urllib.request.Request(url=image_url, headers=headers)
        image_data = urllib.request.urlopen(req).read()
        with open('image.jpg', 'wb') as f:
            f.write(image_data)
        image = Image.open('image.jpg')

        # Get watermark text
        watermark_text = request.form['watermark_text']

        # Resize image for watermark placement
        width, height = image.size
        image_resized = image.resize((500, int(500 * height / width)))

        # Convert image to OpenCV format
        cv_image = cv2.cvtColor(np.array(image_resized), cv2.COLOR_RGB2BGR)

        # Add text watermark to the image
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_thickness = 2
        text_size, _ = cv2.getTextSize(watermark_text, font, font_scale, font_thickness)
        text_x = image_resized.size[0] - text_size[0] - 10
        text_y = image_resized.size[1] - text_size[1] - 10
        cv2.putText(cv_image, watermark_text, (text_x, text_y), font, font_scale, (0, 0, 255), font_thickness, cv2.LINE_AA)

        # Convert the watermarked image back to PIL format
        watermarked_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))

        # Convert the watermarked image to base64
        image_data = io.BytesIO()
        watermarked_image.save(image_data, format='JPEG')
        image_data.seek(0)
        encoded_image = base64.b64encode(image_data.getvalue()).decode('utf-8')

        return render_template('result.html', image=encoded_image)

if __name__ == '__main__':
    app.run(debug=True)
