from flask import Flask, render_template, request, redirect, url_for, send_file, flash
from PIL import Image, ImageDraw
import os
import io
import numpy as np
from scipy.linalg import svd
import zipfile
import base64


app = Flask(__name__)

app = Flask(__name__, template_folder='docs')

app.secret_key = 'your_secret_key_here'

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
BLOCK_SIZE = 450

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def haar_wavelet_transform_2d(image):
    rows, cols = image.shape
    output = np.copy(image).astype(float)
    for r in range(rows):
        temp = np.zeros(cols)
        for i in range(0, cols, 2):
            temp[i//2] = (output[r, i] + output[r, i+1]) / 2
            temp[(cols//2)+(i//2)] = (output[r, i] - output[r, i+1]) / 2
        output[r, :cols] = temp
    for c in range(cols):
        temp = np.zeros(rows)
        for i in range(0, rows, 2):
            temp[i//2] = (output[i, c] + output[i+1, c]) / 2
            temp[(rows//2)+(i//2)] = (output[i, c] - output[i+1, c]) / 2
        output[:rows, c] = temp
    return output


def inverse_haar_wavelet_transform_2d(coeffs):
    rows, cols = coeffs.shape
    output = np.copy(coeffs).astype(float)
    for c in range(cols):
        temp = np.zeros(rows)
        half = rows // 2
        for i in range(half):
            temp[2*i] = coeffs[i, c] + coeffs[half + i, c]
            temp[2*i + 1] = coeffs[i, c] - coeffs[half + i, c]
        output[:rows, c] = temp
    for r in range(rows):
        temp = np.zeros(cols)
        half = cols // 2
        for i in range(half):
            temp[2*i] = output[r, i] + output[r, half + i]
            temp[2*i + 1] = output[r, i] - output[r, half + i]
        output[r, :cols] = temp
    return output


def compress_block_svd(block, k=20):
    U, S, VT = svd(block, full_matrices=False)
    S[k:] = 0
    compressed = np.dot(U, np.dot(np.diag(S), VT))
    return np.clip(compressed, 0, 255).astype(np.uint8)


def compress_block_dwt(block, threshold=10):
    coeffs = haar_wavelet_transform_2d(block)
    rows, cols = coeffs.shape
    for i in range(rows):
        for j in range(cols):
            if i >= rows // 2 or j >= cols // 2:
                if abs(coeffs[i, j]) < threshold:
                    coeffs[i, j] = 0
    compressed = inverse_haar_wavelet_transform_2d(coeffs)
    return np.clip(compressed, 0, 255).astype(np.uint8)


def lossy_compress_block_jpeg(block, quality=30):
    pil_img = Image.fromarray(block)
    with io.BytesIO() as buff:
        pil_img.save(buff, format="JPEG", quality=quality)
        buff.seek(0)
        compressed_img = Image.open(buff).convert("L")
        return np.array(compressed_img)


def block_wise_compress(image_np, roi_blocks, method, k=20, threshold=10, jpeg_quality=30):
    h, w = image_np.shape
    compressed_img = np.zeros_like(image_np)
    num_blocks_vert = h // BLOCK_SIZE
    num_blocks_horz = w // BLOCK_SIZE
    for row in range(num_blocks_vert):
        for col in range(num_blocks_horz):
            y, x = row * BLOCK_SIZE, col * BLOCK_SIZE
            block = image_np[y : y + BLOCK_SIZE, x : x + BLOCK_SIZE]
            if (row, col) in roi_blocks:
                if method == "svd":
                    compressed_block = compress_block_svd(block, k)
                elif method == "dwt":
                    compressed_block = compress_block_dwt(block, threshold)
            else:
                compressed_block = lossy_compress_block_jpeg(block, quality=jpeg_quality)
            compressed_img[y : y + BLOCK_SIZE, x : x + BLOCK_SIZE] = compressed_block
    return compressed_img


def draw_roi_overlay_on_compressed(compressed_np, roi_blocks):
    img_pil = Image.fromarray(compressed_np).convert("RGB")
    draw = ImageDraw.Draw(img_pil, "RGBA")
    h, w = compressed_np.shape
    num_blocks_vert = h // BLOCK_SIZE
    num_blocks_horz = w // BLOCK_SIZE

    for row in range(num_blocks_vert):
        y1 = row * BLOCK_SIZE
        draw.line([(0, y1), (w, y1)], fill="blue", width=2)
    for col in range(num_blocks_horz):
        x1 = col * BLOCK_SIZE
        draw.line([(x1, 0), (x1, h)], fill="blue", width=2)


    for row, col in roi_blocks:
        y, x = row * BLOCK_SIZE, col * BLOCK_SIZE
        draw.rectangle([x, y, x + BLOCK_SIZE, y + BLOCK_SIZE], outline=(255, 0, 0, 255), width=6)
        draw.rectangle([x, y, x + BLOCK_SIZE, y + BLOCK_SIZE], fill=(255, 0, 0, 64))

    return img_pil


def pil_image_to_base64(pil_img, format="PNG"):
    buff = io.BytesIO()
    pil_img.save(buff, format=format)
    return base64.b64encode(buff.getvalue()).decode("utf-8")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/<method>", methods=["GET", "POST"])
def method_page(method):
    method = method.lower()
    if method not in ["dwt", "svd"]:
        return redirect(url_for("home"))

    if request.method == "POST":
        file = request.files.get("image_file")
        roi_data = request.form.get("roi_blocks", "")
        jpeg_quality = int(request.form.get("jpeg_quality", 30))

        if not file or file.filename == "":
            flash("Please select an image file.")
            return redirect(request.url)
        if not roi_data:
            flash("Please select ROI blocks.")
            return redirect(request.url)
        roi_blocks = set(tuple(map(int, p.split(","))) for p in roi_data.strip().split())

        file_bytes = file.read()
        original_size_kb = len(file_bytes) // 1024

        img = Image.open(io.BytesIO(file_bytes)).convert("L")
        w, h = img.size
        w_crop = (w // BLOCK_SIZE) * BLOCK_SIZE
        h_crop = (h // BLOCK_SIZE) * BLOCK_SIZE
        img_np = np.array(img.crop((0, 0, w_crop, h_crop)))

        orig_pil = Image.fromarray(img_np)
        orig_b64 = pil_image_to_base64(orig_pil, format="PNG")

        if method == "svd":
            k = int(request.form.get("k", 20))
            threshold = 10
        else:
            threshold = int(request.form.get("threshold", 10))
            k = 20

        compressed_np = block_wise_compress(img_np, roi_blocks, method, k, threshold, jpeg_quality)

        compressed_roi_img = draw_roi_overlay_on_compressed(compressed_np, roi_blocks)
        compressed_roi_b64 = pil_image_to_base64(compressed_roi_img, format="PNG")

        compressed_img = Image.fromarray(compressed_np)
        compressed_jpeg_path = os.path.join(OUTPUT_FOLDER, f"compressed_{method}.jpg")
        compressed_img.save(compressed_jpeg_path, "JPEG", quality=85, optimize=True)

        compressed_size_kb = os.path.getsize(compressed_jpeg_path) // 1024

        zip_filename = f"compressed_{method}.zip"
        with zipfile.ZipFile(os.path.join(OUTPUT_FOLDER, zip_filename), "w") as zipf:
            zipf.write(compressed_jpeg_path, arcname=f"compressed_{method}.jpg")

        return render_template(
            "success.html",
            zip_filename=zip_filename,
            original_size_kb=original_size_kb,
            compressed_size_kb=compressed_size_kb,
            orig_b64=orig_b64,
            compressed_roi_b64=compressed_roi_b64,
        )

    return render_template("method.html", method=method.upper(), roi_data="")


@app.route("/download_zip/<zip_filename>")
def download_zip(zip_filename):
    zip_path = os.path.join(OUTPUT_FOLDER, zip_filename)
    if os.path.exists(zip_path):
        return send_file(zip_path, as_attachment=True)
    return "File not found.", 404


if __name__ == "__main__":
    app.run(debug=True)
