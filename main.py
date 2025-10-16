from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import cv2 as cv
from io import BytesIO
import numpy as np
import os
from waitress import serve


app = Flask(__name__)
CORS(app)


def grayscale_by_technique(image_bgr: np.ndarray, technique: str, *, weighted_weights: tuple[float, float, float] | None = None, single_channel: str | None = None) -> np.ndarray:
    """
    Técnicas suportadas: average, luminosity, lightness, desaturation, single_channel(r|g|b), weighted.

    """
    tech = (technique or "").strip().lower()

    if tech in ("average", "avg"):
        gray = image_bgr.mean(axis=2)
        return gray.astype(np.uint8)

    if tech in ("luminosity", "luma", "bt709"):
        # pesos perceptuais BT.709 (para RGB). Como a imagem é BGR, invertimos a ordem
        weights_bgr = np.array([0.0722, 0.7152, 0.2126], dtype=np.float32)
        gray = (image_bgr.astype(np.float32) * weights_bgr).sum(axis=2)
        return np.clip(gray, 0, 255).astype(np.uint8)

    if tech == "lightness":
        mx = image_bgr.max(axis=2).astype(np.float32)
        mn = image_bgr.min(axis=2).astype(np.float32)
        gray = (mx + mn) / 2.0
        return gray.astype(np.uint8)

    if tech == "desaturation":
        mx = image_bgr.max(axis=2).astype(np.float32)
        mn = image_bgr.min(axis=2).astype(np.float32)
        gray = (mx + mn) / 2.0
        return gray.astype(np.uint8)

    if tech == "single_channel":
        ch = (single_channel or "b").lower()
        if ch == "r":
            return image_bgr[:, :, 2]
        if ch == "g":
            return image_bgr[:, :, 1]
        return image_bgr[:, :, 0]

    if tech == "weighted":
        wr, wg, wb = weighted_weights or (0.2126, 0.7152, 0.0722)
        # pesos estão no espaço RGB; imagem está em BGR → reordenar para B,G,R
        weights_bgr = np.array([wb, wg, wr], dtype=np.float32)
        gray = (image_bgr.astype(np.float32) * weights_bgr).sum(axis=2)
        return np.clip(gray, 0, 255).astype(np.uint8)

    # fallback para luminosity (bt709)
    weights_bgr = np.array([0.0722, 0.7152, 0.2126], dtype=np.float32)
    gray = (image_bgr.astype(np.float32) * weights_bgr).sum(axis=2)
    return np.clip(gray, 0, 255).astype(np.uint8)

@app.route('/grayscale', methods=['POST'])
def grayscale():
    if 'image' not in request.files:
        return jsonify({'error': 'missing file field "image"'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'empty filename'}), 400
    
    try:
          # Lê o arquivo em bytes
        file_bytes = np.frombuffer(file.read(), np.uint8)

        # Decodifica a imagem usando OpenCV
        image = cv.imdecode(file_bytes, cv.IMREAD_COLOR)
        if image is None:
            return jsonify({'error': 'invalid or unsupported image'}), 400

        # Técnica via query (?technique=average|luminosity|lightness|desaturation|single_channel|weighted)
        technique = request.args.get('technique') or request.form.get('technique')
        # Param extra para single_channel (r|g|b)
        single_ch = request.args.get('channel') or request.form.get('channel')
        # Pesos para weighted (wr,wg,wb)
        weights_q = request.args.get('weights') or request.form.get('weights')
        custom_weights = None
        if weights_q:
            try:
                wr, wg, wb = [float(x) for x in weights_q.split(',')]
                custom_weights = (wr, wg, wb)
            except Exception:
                return jsonify({'error': 'invalid weights format. Use wr,wg,wb'}), 400

        # Converte conforme técnica selecionada
        grayscale_image = grayscale_by_technique(
            image,
            technique,
            weighted_weights=custom_weights,
            single_channel=single_ch,
        )

         # Codifica novamente para PNG em memória
        success, buffer = cv.imencode('.png', grayscale_image)
        buf = BytesIO(buffer)
        if not success:
            return jsonify({'error': 'failed to encode image'}), 500

        return send_file(buf, mimetype='image/png')

    except Exception:
        return jsonify({'error': 'failed to process image'}), 500


@app.route('/filter', methods=['POST'])
def apply_filter():
    if 'image' not in request.files:
        return jsonify({'error': 'missing file field "image"'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'empty filename'}), 400

    try:
        file_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv.imdecode(file_bytes, cv.IMREAD_COLOR)
        if image is None:
            return jsonify({'error': 'invalid or unsupported image'}), 400

        ftype = (request.args.get('type') or request.form.get('type') or '').strip().lower()

        if ftype == 'gaussian':
            ksize = int(request.args.get('ksize') or request.form.get('ksize') or 5)
            if ksize % 2 == 0 or ksize < 3:
                return jsonify({'error': 'ksize must be odd and >=3'}), 400
            sigma = float(request.args.get('sigma') or request.form.get('sigma') or 0)
            output = cv.GaussianBlur(image, (ksize, ksize), sigma)

        elif ftype == 'median':
            ksize = int(request.args.get('ksize') or request.form.get('ksize') or 5)
            if ksize % 2 == 0 or ksize < 3:
                return jsonify({'error': 'ksize must be odd and >=3'}), 400
            output = cv.medianBlur(image, ksize)

        elif ftype == 'sharpen':
            kernel = np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]], dtype=np.float32)
            output = cv.filter2D(image, -1, kernel)

        elif ftype == 'sobel':
            ksize = int(request.args.get('ksize') or request.form.get('ksize') or 3)
            axis = (request.args.get('axis') or request.form.get('axis') or 'xy').lower()
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            sobelx = cv.Sobel(gray, cv.CV_64F, 1 if axis in ('x','xy') else 0, 0, ksize=ksize)
            sobely = cv.Sobel(gray, cv.CV_64F, 0, 1 if axis in ('y','xy') else 0, ksize=ksize)
            mag = np.sqrt((sobelx ** 2) + (sobely ** 2)) if axis == 'xy' else (np.abs(sobelx) if axis == 'x' else np.abs(sobely))
            mag = (mag / (mag.max() + 1e-8) * 255).astype(np.uint8)
            output = cv.cvtColor(mag, cv.COLOR_GRAY2BGR)

        elif ftype == 'canny':
            th1 = int(request.args.get('th1') or request.form.get('th1') or 100)
            th2 = int(request.args.get('th2') or request.form.get('th2') or 200)
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            edges = cv.Canny(gray, th1, th2)
            output = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)

        else:
            return jsonify({'error': 'invalid filter type. Use gaussian|median|sharpen|sobel|canny'}), 400

        success, buffer = cv.imencode('.png', output)
        if not success:
            return jsonify({'error': 'failed to encode image'}), 500
        buf = BytesIO(buffer)
        return send_file(buf, mimetype='image/png')

    except Exception:
        return jsonify({'error': 'failed to process image'}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    serve(app, host='0.0.0.0', port=port)