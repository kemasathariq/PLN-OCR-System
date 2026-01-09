import os
import shutil
from ultralytics import YOLO
import cv2
import numpy as np
import easyocr
import re
import torch

DEVICE = 0 if torch.cuda.is_available() else 'cpu'

# =======================
# KONFIGURASI
# =======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(BASE_DIR, "train2", "weights", "best.pt"))  # path model YOLO
INPUT_FOLDER = os.environ.get("INPUT_FOLDER", os.path.join(BASE_DIR, "input_images"))
OUTPUT_FOLDER = os.environ.get("OUTPUT_FOLDER", os.path.join(BASE_DIR, "output"))
DEBUG_FOLDER = os.path.join(OUTPUT_FOLDER, "debug_crops")
STAND_DIGITS = 5          # jumlah digit stand yang ingin dibaca (ubah ke 5 jika tanpa roda merah)
CUT_RIGHT_STAND = 0.12    # porsi kanan yang dipotong untuk buang roda merah
CENTER_BAND = (0.25, 0.80)  # ambil pita vertikal 25%..80% dari tinggi
SAVE_DEBUG_CROPS = os.environ.get("SAVE_DEBUG_CROPS", "1") == "1"

# Jalankan OCR hanya pada kelas berikut
TEXT_CLASSES = {"stand meter", "nomor meter"}

# Stand meter umumnya angka saja
ALLOW_DIGITS_ONLY = True

# Simpan crop ROI untuk debug (True/False)

def print_gpu_info():
    if torch.cuda.is_available():
        print(f"[GPU] CUDA {torch.version.cuda} | {torch.cuda.get_device_name(0)}")
    else:
        print("[GPU] CUDA TIDAK TERSEDIA — jatuh ke CPU")

# panggil sekali di awal detect_objects()
print_gpu_info()
# =======================
# UTIL FOLDER
# =======================
def ensure_dirs() -> bool:
    if not os.path.exists(INPUT_FOLDER):
        os.makedirs(INPUT_FOLDER)
        print(f"Folder '{INPUT_FOLDER}' dibuat. Masukkan gambar (.png/.jpg/.jpeg) ke folder ini lalu jalankan ulang.")
        return False

    if os.path.exists(OUTPUT_FOLDER):
        shutil.rmtree(OUTPUT_FOLDER)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    return True


# =======================
# CROP & PRE-PROCESS
# =======================
def safe_crop(image, xyxy, pad=18):
    """Crop ROI dengan padding aman."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = map(int, xyxy)
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)
    return image[y1:y2, x1:x2].copy()

def refine_crop_by_class(cls_name, crop):
    """
    Persempit ROI agar OCR fokus:
    - stand meter: buang sisi kanan (disc wheel merah) + center strip vertikal.
    - nomor meter: trim tipis tepi agar glare/tepi stiker hilang.
    """
    h, w = crop.shape[:2]
    if h < 8 or w < 8:
        return crop

    if cls_name == "stand meter":
        # ambil 80–85% kiri (hindari roda merah) + 60% tengah vertikal
        x2 = int(0.85 * w)
        y1 = int(0.20 * h)
        y2 = int(0.80 * h)
        crop = crop[y1:y2, 0:x2]

    elif cls_name == "nomor meter":
        # potong pinggir tipis supaya semua digit masuk tanpa tepi stiker
        x1 = int(0.03 * w); x2 = int(0.97 * w)
        y1 = int(0.15 * h); y2 = int(0.85 * h)
        crop = crop[y1:y2, x1:x2]

    return crop

def rotate_bound(image, angle):
    """Rotate image while keeping all pixels in view."""
    h, w = image.shape[:2]
    center = (w / 2, h / 2)
    m = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = abs(m[0, 0]); sin = abs(m[0, 1])
    n_w = int((h * sin) + (w * cos))
    n_h = int((h * cos) + (w * sin))
    m[0, 2] += (n_w / 2) - center[0]
    m[1, 2] += (n_h / 2) - center[1]
    return cv2.warpAffine(image, m, (n_w, n_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def estimate_skew_angle(bgr):
    """Estimasi kemiringan dominan ROI (derajat)."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=max(30, int(0.12 * min(gray.shape[:2]))))
    if lines is None:
        return 0.0

    angles = []
    for line in lines:
        rho, theta = line[0]
        deg = theta * 180.0 / np.pi
        if deg > 90:
            deg -= 180  # ubah ke range [-90, 90]
        if -60 <= deg <= 60:  # fokus ke garis hampir horizontal
            angles.append(deg)

    if not angles:
        return 0.0
    return float(np.median(angles))

def auto_orient_crop(crop):
    """Perbaiki orientasi crop (auto rotate 90° jika portrait & deskew kecil)."""
    if crop is None or crop.size == 0:
        return crop

    oriented = crop
    h, w = oriented.shape[:2]

    # Jika crop terlalu tinggi (kemungkinan foto sideways), coba rotasi 90°
    if h > w * 1.15:
        oriented = rotate_bound(oriented, 90)
        h, w = oriented.shape[:2]

    # Koreksi kemiringan kecil dengan Hough
    skew = estimate_skew_angle(oriented)
    if abs(skew) > 0.8:  # abaikan noise <1 derajat
        oriented = rotate_bound(oriented, -skew)

    return oriented

def fix_similar_chars(s: str) -> str:
    # ubah huruf mirip digit -> digit
    tbl = str.maketrans({
        'O': '0', 'o': '0', 'Q': '0',
        'I': '1', 'l': '1', '|': '1',
        'B': '8',
        'S': '5',
        'Z': '2'
    })
    return s.translate(tbl)

def prep_variants(bgr, boost=True, try_invert=True):
    """Buat beberapa varian pre-proses untuk meningkatkan keberhasilan OCR."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    if boost:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

    # Tajamkan digit kecil
    blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
    sharp = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)

    # Threshold adaptif (sering membantu label biru/gelap)
    thr = cv2.adaptiveThreshold(
        sharp, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 8
    )

    variants = [gray, sharp, thr]
    if try_invert:
        variants += [255 - v for v in variants]
    return variants


# =======================
# OCR
# =======================
def safe_readtext(reader, img, **kwargs):
    """Wrapper aman untuk EasyOCR.readtext agar tidak crash bila ROI terlalu kecil/kosong."""
    try:
        if img is None:
            return []
        if img.size == 0:
            return []
        if len(img.shape) == 2:
            h, w = img.shape
        else:
            h, w = img.shape[:2]
        # minimal size agar internal EasyOCR tidak memotong jadi 0
        if h < 8 or w < 8:
            return []
        if img.dtype != np.uint8:
            img = img.astype(np.uint8, copy=False)
        return reader.readtext(img, **kwargs)
    except Exception:
        return []

def ocr_best(reader, crop_bgr, digits_only=True, scales=(1.5, 2.0, 3.0), target_len=None):
    allowlist = '0123456789' if digits_only else None
    best_text, best_score = "", -1.0

    if crop_bgr is None or crop_bgr.size == 0 or min(crop_bgr.shape[:2]) < 5:
        return "", -1.0

    for s in scales:
        resized = cv2.resize(crop_bgr, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
        for img in prep_variants(resized, boost=True, try_invert=True):
            results = safe_readtext(
                reader, img,
                detail=1, paragraph=False, allowlist=allowlist,
                low_text=0.25, text_threshold=0.3, link_threshold=0.3
        )

            if not results:
                continue

            # gabung semua potongan
            texts = [t.strip() for (_b, t, _c) in results if t and t.strip()]
            confs = [float(_c) for (_b, _t, _c) in results if _t and _t.strip()]
            if not texts:
                continue

            joined = " ".join(texts)
            avg_conf = float(np.mean(confs)) if confs else 0.0

            # skor = confidence + bonus kedekatan panjang (jika target_len diset)
            length_bonus = 0.0
            if target_len is not None:
                length_bonus = max(0.0, 1.0 - abs(len(re.sub(r'\\s+', '', joined)) - target_len) / max(1, target_len))
            score = avg_conf + 0.2 * length_bonus  # bobot 0.2 bisa diubah

            if score > best_score:
                best_score, best_text = score, joined

    return best_text.strip(), best_score

def postprocess_text(text, digits_only=True):
    if not text:
        return text
    text = fix_similar_chars(text)  # normalisasi dulu
    if digits_only:
        return re.sub(r"[^0-9]", "", text)
    return re.sub(r"[^0-9A-Za-z]", "", text).upper()


def group_easyocr_lines(results, y_tol=18):
    """Group EasyOCR results into text lines (top-to-bottom) and return list of lines sorted left-to-right."""
    items = []
    for bbox, text, conf in results:
        if not text or not str(text).strip():
            continue
        ys = [p[1] for p in bbox]
        xs = [p[0] for p in bbox]
        yc = float(sum(ys)) / len(ys)
        xc = float(sum(xs)) / len(xs)
        items.append((yc, xc, str(text), float(conf)))
    if not items:
        return []
    items.sort(key=lambda x: x[0])
    lines = []
    cur = [items[0]]
    for it in items[1:]:
        if abs(it[0] - cur[-1][0]) <= y_tol:
            cur.append(it)
        else:
            lines.append(cur)
            cur = [it]
    lines.append(cur)
    out = []
    for line in lines:
        line.sort(key=lambda x: x[1])
        text_join = "".join([t for (_yc,_xc,t,_c) in line])
        conf_avg = float(np.mean([c for (_yc,_xc,_t,c) in line]))
        out.append({"text": text_join, "conf": conf_avg})
    return out

def pick_best_digit_window(digits, target_lengths):
    """If digits is too long, pick the best substring whose length is in target_lengths (prefer exact)."""
    digits = re.sub(r"[^0-9]", "", digits or "")
    if not digits:
        return ""
    for L in target_lengths:
        if len(digits) == L:
            return digits
    # if longer, sliding windows
    if len(digits) > max(target_lengths):
        best = ""
        best_score = -1e9
        for L in target_lengths:
            if len(digits) < L:
                continue
            for i in range(0, len(digits) - L + 1):
                cand = digits[i:i+L]
                # simple heuristic: avoid windows with too many repeated chars (often noise)
                rep_penalty = max(cand.count(str(d)) for d in range(10)) / L
                score = 1.0 - rep_penalty
                if score > best_score:
                    best, best_score = cand, score
        return best or digits
    # if shorter, return as is
    return digits

def ocr_digit_single(reader, patch):
    # OCR untuk satu digit (patch kecil), pakai skala kuat
    txt, _ = ocr_best(reader, patch, digits_only=True, scales=(2.0, 3.0, 4.0))
    return postprocess_text(txt, digits_only=True)[:1]  # ambil 1 char saja

def score_length_match(text, target_len, tol=1):
    """Skor sederhana berdasar kedekatan panjang teks dengan target."""
    if not text:
        return -1.0
    return max(0.0, 1.0 - abs(len(text) - target_len) / max(target_len, tol))

def ocr_stand_by_slices(reader, crop_bgr, k=STAND_DIGITS):
    """
    Bagi ROI stand menjadi k kolom dan OCR satu per satu.
    Lebih robust dibanding OCR seluruh baris sekaligus.
    """
    h, w = crop_bgr.shape[:2]
    # buang sisi kanan (roda merah), ambil pita tengah vertikal
    x2 = int((1.0 - CUT_RIGHT_STAND) * w)
    y1 = int(CENTER_BAND[0] * h)
    y2 = int(CENTER_BAND[1] * h)
    band = crop_bgr[y1:y2, 0:x2]
    if band.size == 0:
        return ""

    # bagi jadi k kolom sama lebar
    bw = band.shape[1] // k if k else band.shape[1]
    digits = []
    for i in range(k):
        x1s = i * bw
        x2s = (i + 1) * bw if i < k - 1 else band.shape[1]
        patch = band[:, x1s:x2s]
        # upscale dikit
        patch = cv2.resize(patch, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
        d = ocr_digit_single(reader, patch)
        digits.append(d if d else "")
    return "".join(digits)

def ocr_stand_meter(reader, crop_bgr, k=STAND_DIGITS):
    """
    Stand meter: coba OCR 1-baris dulu (lebih stabil kalau crop sudah tepat).
    Kalau gagal/hasil tidak masuk akal -> fallback slice-per-digit.
    """
    # 1) OCR baris
    txt, score = ocr_best(reader, crop_bgr, digits_only=True, target_len=k, scales=(2.0, 3.0, 4.0))
    digits = postprocess_text(txt, digits_only=True)
    # buang digit merah (0.1) kalau ikut kebaca; ambil window terbaik panjang k / k+1
    cand = pick_best_digit_window(digits, target_lengths=(k, k+1))
    if len(cand) == k:
        return cand
    # 2) fallback ke slicing
    sliced = ocr_stand_by_slices(reader, crop_bgr, k=k)
    sliced = postprocess_text(sliced, digits_only=True)
    cand2 = pick_best_digit_window(sliced, target_lengths=(k,))
    return cand2[:k]

def ocr_nomor_meter(reader, crop_bgr, expected_len_range=(9, 13)):
    """
    OCR nomor meter (serial).

    Problem lama: hasil EasyOCR per-token digabung semua -> jadi angka super panjang.
    Solusi: group per-baris, lalu pilih kandidat digit terbaik (panjang masuk rentang, conf tinggi),
    dan jika terlalu panjang ambil window terbaik di rentang panjang.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return ""

    min_len, max_len = expected_len_range
    target_lengths = tuple(range(min_len, max_len + 1))

    best_text = ""
    best_score = -1e9

    # agresif: multi-scale + beberapa variant preprocess
    for s in (2.5, 3.0, 4.0, 5.0):
        resized = cv2.resize(crop_bgr, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
        for img in prep_variants(resized, boost=True, try_invert=True):
            results = safe_readtext(
                reader, img,
                detail=1, paragraph=False, allowlist="0123456789",
                low_text=0.25, text_threshold=0.3, link_threshold=0.3
            )
            if not results:
                continue

            for line in group_easyocr_lines(results, y_tol=22):
                digits = postprocess_text(line["text"], digits_only=True)
                if not digits:
                    continue
                # windowing kalau kepanjangan
                cand = pick_best_digit_window(digits, target_lengths)

                # scoring: prefer length in range + conf tinggi
                length_ok = (min_len <= len(cand) <= max_len)
                length_bonus = 1.0 if length_ok else -abs(len(cand) - (min_len+max_len)//2) * 0.15
                score = float(line["conf"]) * 3.0 + length_bonus

                if score > best_score:
                    best_score = score
                    best_text = cand

    return best_text
def ocr_with_orientation(reader, cls_name, crop_bgr, debug_dir=None, tag=""):
    """
    Coba beberapa orientasi (0/90/-90/180) lalu pilih OCR terbaik.
    Skor berdasarkan hasil OCR (confidence atau panjang mendekati target).
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return "", 0


    # Debug saving (crop sebelum/ sesudah orientasi & refine)
    if SAVE_DEBUG_CROPS and debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        if tag:
            cv2.imwrite(os.path.join(debug_dir, f"{tag}_00_input.jpg"), crop_bgr)
    base = auto_orient_crop(crop_bgr)
    if base is None or base.size == 0:
        return "", 0

    if SAVE_DEBUG_CROPS and debug_dir and tag:
        cv2.imwrite(os.path.join(debug_dir, f"{tag}_01_base.jpg"), base)

    rotations = [0, 180]
    if base.shape[0] > base.shape[1] * 1.05:
        rotations = [0, 90, -90, 180]

    best_txt, best_score, best_angle = "", -1.0, 0

    for ang in rotations:
        rotated = rotate_bound(base, ang) if ang != 0 else base
        refined = refine_crop_by_class(cls_name, rotated)
        if refined is None or refined.size == 0:
            continue

        if SAVE_DEBUG_CROPS and debug_dir and tag:
            cv2.imwrite(os.path.join(debug_dir, f"{tag}_ang{ang:+d}_{cls_name}.jpg"), refined)

        if cls_name == "stand meter":
            txt = ocr_stand_meter(reader, refined, k=STAND_DIGITS)
            score = score_length_match(txt, STAND_DIGITS, tol=1)
        elif cls_name == "nomor meter":
            txt = ocr_nomor_meter(reader, refined, expected_len_range=(9, 13))
            # Skor 1 jika panjang dalam rentang, turun jika jauh.
            if 9 <= len(txt) <= 13:
                score = 1.0
            else:
                near = min(abs(len(txt) - 9), abs(len(txt) - 13))
                score = max(0.0, 1.0 - near / 13.0)
        else:
            txt_raw, score_raw = ocr_best(reader, refined, digits_only=ALLOW_DIGITS_ONLY, scales=(1.5, 2.0, 3.0))
            txt = postprocess_text(txt_raw, digits_only=ALLOW_DIGITS_ONLY)
            score = score_raw

        if score > best_score:
            best_txt, best_score, best_angle = txt, score, ang

    return best_txt, best_angle


# =======================
# PIPELINE UTAMA
# =======================
def draw_ocr_on_box(img, box_xyxy, text, bg=(0, 255, 255), fg=(0, 0, 0)):
    """
    Gambar kotak label berisi teks OCR di dekat bounding box.
    - img: BGR image
    - box_xyxy: (x1, y1, x2, y2)
    - text: string yang akan ditaruh, mis. 'stand meter: 014012'
    - bg: warna latar (BGR), default kuning
    - fg: warna teks (BGR), default hitam
    """
    if not text:
        return

    x1, y1, x2, y2 = [int(v) for v in box_xyxy]
    h_img = img.shape[0]

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thick = 2
    pad = 4

    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)

    # Posisi label: coba di bawah box; kalau mentok, taruh di atas
    y_bot = y2 + th + 2 * pad
    if y_bot <= h_img:
        top_left = (x1, y_bot - th - 2 * pad)
        bottom_right = (x1 + tw + 2 * pad, y_bot)
        org = (x1 + pad, y_bot - pad)
    else:
        y_top = max(0, y1 - th - 2 * pad)
        top_left = (x1, y_top)
        bottom_right = (x1 + tw + 2 * pad, y_top + th + 2 * pad)
        org = (x1 + pad, y_top + th + pad)

    cv2.rectangle(img, top_left, bottom_right, bg, -1)
    cv2.putText(img, text, org, font, scale, fg, thick, cv2.LINE_AA)


def detect_objects():
    if not ensure_dirs():
        return

    # Load YOLO
    model = YOLO(MODEL_PATH)

    # EasyOCR: 'en' cukup untuk angka. Set gpu=True kalau CUDA siap & stabil.
    reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())

    # Daftar gambar
    image_files = [
        f for f in os.listdir(INPUT_FOLDER)
        if os.path.isfile(os.path.join(INPUT_FOLDER, f))
        and f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]
    if not image_files:
        print(f"Tidak ada file gambar di '{INPUT_FOLDER}'. Tambahkan .png/.jpg/.jpeg lalu jalankan ulang.")
        return

    # Mapping id->nama kelas (jika ada)
    try:
        names = model.model.names if hasattr(model, "model") and hasattr(model.model, "names") else None
    except Exception:
        names = None

    for img_name in image_files:
        input_path = os.path.join(INPUT_FOLDER, img_name)
        output_path_img = os.path.join(OUTPUT_FOLDER, img_name)
        output_path_txt = os.path.join(OUTPUT_FOLDER, os.path.splitext(img_name)[0] + "_ocr.txt")
        dbg_dir = os.path.join(DEBUG_FOLDER, os.path.splitext(img_name)[0])
        if SAVE_DEBUG_CROPS:
            os.makedirs(dbg_dir, exist_ok=True)

        # Inference YOLO
        results = model.predict(
            source=input_path,
            device=DEVICE,
            save=False,
            conf=0.25,
            iou=0.45,
            max_det=1000,
            verbose=False
        )

        image_bgr = cv2.imread(input_path)
        ocr_lines = []

        for result in results:
            # ambil gambar beranotasi YOLO dulu, tapi JANGAN disimpan dulu
            annotated_image = result.plot()

            if not hasattr(result, "boxes") or result.boxes is None:
                cv2.imwrite(output_path_img, annotated_image)
                continue

            boxes = result.boxes
            xyxy = boxes.xyxy.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else np.zeros((len(xyxy),), dtype=int)
            confs = boxes.conf.cpu().numpy() if boxes.conf is not None else np.ones((len(xyxy),), dtype=float)

            for i in range(len(xyxy)):
                cls_id = clss[i] if i < len(clss) else -1
                det_conf = float(confs[i]) if i < len(confs) else 0.0
                box = xyxy[i]
                # resolve class name (Ultralytics: names bisa dict atau list)
                if isinstance(names, dict):
                    cls_name = names.get(cls_id, str(cls_id))
                elif isinstance(names, (list, tuple)) and 0 <= cls_id < len(names):
                    cls_name = names[cls_id]
                else:
                    cls_name = str(cls_id)

                if TEXT_CLASSES and cls_name not in TEXT_CLASSES:
                    continue

                crop = safe_crop(image_bgr, box, pad=18)
                if crop.size == 0:
                    continue
                if SAVE_DEBUG_CROPS:
                    cv2.imwrite(os.path.join(dbg_dir, f"{i:02d}_{cls_name}_raw.jpg"), crop)

                text_clean, used_angle = ocr_with_orientation(reader, cls_name, crop, debug_dir=dbg_dir, tag=f"{i:02d}_{cls_name}")

                if text_clean:
                    # 1) simpan ke txt
                    x1, y1, x2, y2 = map(int, box)
                    ocr_lines.append(
                        f"[{cls_name} det_conf={det_conf:.2f} box=({x1},{y1},{x2},{y2}) angle={used_angle}deg] OCR='{text_clean}'"
                    )

                    # 2) gambar teks OCR ke gambar anotasi
                    draw_ocr_on_box(
                        annotated_image,
                        box,
                        f"{cls_name}: {text_clean}",
                        bg=(0, 255, 255),   # kuning
                        fg=(0, 0, 0)        # hitam
                    )

            # simpan gambar SETELAH menempelkan teks OCR
            cv2.imwrite(output_path_img, annotated_image)

                # --- OCR END ---

        # Tulis hasil OCR
        if ocr_lines:
            with open(output_path_txt, "w", encoding="utf-8") as f:
                f.write("\n".join(ocr_lines))
        else:
            with open(output_path_txt, "w", encoding="utf-8") as f:
                f.write("(Tidak ada teks terbaca dari deteksi yang dipilih.)\n")

        print(f"Hasil deteksi disimpan ke: {output_path_img}")
        print(f"Hasil OCR disimpan ke   : {output_path_txt}")

    print("Proses deteksi & OCR selesai untuk semua gambar.")

# =======================
# MAIN
# =======================
if __name__ == "__main__":
    detect_objects()
