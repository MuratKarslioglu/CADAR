import cv2
import numpy as np
import json
import argparse
from pathlib import Path

PROFILE_PATH = Path("profile.json")


def odd(n: int) -> int:
    return n + 1 if n % 2 == 0 else n


def nothing(_=None):
    pass


def build_trackbars():
    cv2.namedWindow("Controls", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Controls", 400, 500)

    # HSV ranges
    cv2.createTrackbar("H low", "Controls", 0, 179, nothing)
    cv2.createTrackbar("H high", "Controls", 179, 179, nothing)
    cv2.createTrackbar("S low", "Controls", 0, 255, nothing)
    cv2.createTrackbar("S high", "Controls", 255, 255, nothing)
    cv2.createTrackbar("V low", "Controls", 0, 255, nothing)
    cv2.createTrackbar("V high", "Controls", 255, 255, nothing)

    # Pre-proc & edges
    cv2.createTrackbar("Blur (0-25)", "Controls", 3, 25,
                       nothing)      # kernel = odd(val)
    cv2.createTrackbar("Canny th1", "Controls", 100, 500, nothing)
    cv2.createTrackbar("Canny th2", "Controls", 200, 500, nothing)

    # Morphology
    cv2.createTrackbar("Kernel (1-25)", "Controls", 5, 25, nothing)    # odd
    cv2.createTrackbar("Morph iters", "Controls", 1, 10,
                       nothing)      # +/- iterations
    cv2.createTrackbar("Use close (1) / open (0)", "Controls", 1, 1, nothing)


def read_trackbar_params():
    hl = cv2.getTrackbarPos("H low", "Controls")
    hh = cv2.getTrackbarPos("H high", "Controls")
    sl = cv2.getTrackbarPos("S low", "Controls")
    sh = cv2.getTrackbarPos("S high", "Controls")
    vl = cv2.getTrackbarPos("V low", "Controls")
    vh = cv2.getTrackbarPos("V high", "Controls")
    blur = odd(max(0, cv2.getTrackbarPos("Blur (0-25)", "Controls")))
    c1 = cv2.getTrackbarPos("Canny th1", "Controls")
    c2 = cv2.getTrackbarPos("Canny th2", "Controls")
    k = odd(max(1, cv2.getTrackbarPos("Kernel (1-25)", "Controls")))
    iters = cv2.getTrackbarPos("Morph iters", "Controls")
    use_close = cv2.getTrackbarPos("Use close (1) / open (0)", "Controls") == 1

    return {
        "hsv_low": [hl, sl, vl],
        "hsv_high": [hh, sh, vh],
        "blur": blur,
        "canny": [c1, c2],
        "kernel": k,
        "iters": iters,
        "use_close": use_close
    }


def apply_pipeline(frame, params):
    """Returns (mask, masked_bgr, edges)"""
    img = frame.copy()
    if params["blur"] > 0:
        img = cv2.GaussianBlur(img, (params["blur"], params["blur"]), 0)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    low = np.array(params["hsv_low"], dtype=np.uint8)
    high = np.array(params["hsv_high"], dtype=np.uint8)
    mask = cv2.inRange(hsv, low, high)

    # Morphology
    k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (params["kernel"], params["kernel"]))
    if params["iters"] > 0:
        if params["use_close"]:
            mask = cv2.morphologyEx(
                mask, cv2.MORPH_CLOSE, k, iterations=params["iters"])
        else:
            mask = cv2.morphologyEx(
                mask, cv2.MORPH_OPEN, k, iterations=params["iters"])

    # Edges (informative)
    edges = cv2.Canny(img, params["canny"][0], params["canny"][1])

    masked = cv2.bitwise_and(frame, frame, mask=mask)
    return mask, masked, edges


def stack_horiz(imgs, max_height=480):
    resized = []
    for im in imgs:
        if im.ndim == 2:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        h, w = im.shape[:2]
        scale = max_height / h
        im = cv2.resize(im, (int(w * scale), int(h * scale)))
        resized.append(im)
    return cv2.hconcat(resized)


def save_profile(params, view_mode):
    data = {"params": params, "view_mode": view_mode}
    with open(PROFILE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"[i] Ayarlar kaydedildi -> {PROFILE_PATH.resolve()}")


def load_profile():
    if PROFILE_PATH.exists():
        with open(PROFILE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"[i] Ayarlar yüklendi <- {PROFILE_PATH.resolve()}")
        return data.get("params"), data.get("view_mode", 3)
    return None, 3


def set_trackbars_from_params(p):
    cv2.setTrackbarPos("H low", "Controls", int(p["hsv_low"][0]))
    cv2.setTrackbarPos("S low", "Controls", int(p["hsv_low"][1]))
    cv2.setTrackbarPos("V low", "Controls", int(p["hsv_low"][2]))
    cv2.setTrackbarPos("H high", "Controls", int(p["hsv_high"][0]))
    cv2.setTrackbarPos("S high", "Controls", int(p["hsv_high"][1]))
    cv2.setTrackbarPos("V high", "Controls", int(p["hsv_high"][2]))
    cv2.setTrackbarPos("Blur (0-25)", "Controls", int(p["blur"]))
    cv2.setTrackbarPos("Canny th1", "Controls", int(p["canny"][0]))
    cv2.setTrackbarPos("Canny th2", "Controls", int(p["canny"][1]))
    cv2.setTrackbarPos("Kernel (1-25)", "Controls", int(p["kernel"]))
    cv2.setTrackbarPos("Morph iters", "Controls", int(p["iters"]))
    cv2.setTrackbarPos(
        "Use close (1) / open (0)",
        "Controls",
        1 if p["use_close"] else 0)


def make_tracker():
    tracker = None
    # Try CSRT then KCF
    try:
        tracker = cv2.legacy.TrackerCSRT_create()
    except Exception:
        try:
            tracker = cv2.TrackerCSRT_create()
        except Exception:
            pass
    if tracker is None:
        try:
            tracker = cv2.legacy.TrackerKCF_create()
        except Exception:
            try:
                tracker = cv2.TrackerKCF_create()
            except Exception:
                pass
    return tracker


def main():
    parser = argparse.ArgumentParser(
        description="Interactive Object Reveal & Tracking UI (OpenCV)")
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Video dosya yolu (boşsa kamera kullanılacak)")
    parser.add_argument("--camera", type=int, default=0, help="Kamera indexi")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video if args.video else args.camera)
    if not cap.isOpened():
        print("[!] Kaynak açılamadı.")
        return

    build_trackbars()
    default_params = {
        "hsv_low": [0, 30, 30],
        "hsv_high": [179, 255, 255],
        "blur": 3,
        "canny": [100, 200],
        "kernel": 5,
        "iters": 1,
        "use_close": True
    }
    set_trackbars_from_params(default_params)

    # View mode: 1=original, 2=mask, 3=result
    view_mode = 3

    tracker = None
    has_tracker = False
    bbox = None

    # play/pause
    paused = False
    last_frame = None

    # Yükleme varsa uygula
    loaded_params, loaded_view = load_profile()
    if loaded_params:
        set_trackbars_from_params(loaded_params)
        view_mode = loaded_view

    cv2.namedWindow("View", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("View", 1280, 520)

    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                # Video dosyasının sonu -> yeniden aç ve devam et (loop)
                if args.video:
                    cap.release()
                    cap = cv2.VideoCapture(args.video)
                    ok, frame = cap.read()
                    if not ok:
                        print("[!] Video yeniden baslatilamadi.")
                        break
                else:
                    # Kamera ise okuma basarisizsa cik
                    print("[!] Kamera akisi kesildi.")
                    break
            last_frame = frame.copy()
        else:
            # Duraklatilmis: son kareyi kullan
            if last_frame is None:
                ok, frame = cap.read()
                if not ok:
                    break
                last_frame = frame.copy()
            frame = last_frame.copy()

        params = read_trackbar_params()
        mask, masked, edges = apply_pipeline(frame, params)

        # Tracker
        if has_tracker and tracker is not None and not paused:
            ok, bbox = tracker.update(frame)
            if ok:
                (x, y, w, h) = [int(v) for v in bbox]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, "TRACKING", (x, max(0, y - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "LOST", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        # Görünüm paneli
        panel = stack_horiz([frame, mask, masked], max_height=480)

        # NEW: Duraklatma etiketi
        if paused:
            cv2.putText(panel, "PAUSED", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

        cv2.imshow("View", panel)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break
        elif key == ord(' '):  # Space: pause/resume
            paused = not paused
        elif key == ord('n'):  # Next frame when paused
            # Sadece bir kare ilerlemek icin paused modunda bir kez oku
            if paused and args.video:
                ok, frame = cap.read()
                if not ok:
                    # Videonun sonu: loop ve bir kare al
                    cap.release()
                    cap = cv2.VideoCapture(args.video)
                    ok, frame = cap.read()
                if ok:
                    last_frame = frame.copy()
        elif key == ord('1'):
            view_mode = 1
        elif key == ord('2'):
            view_mode = 2
        elif key == ord('3'):
            view_mode = 3
        elif key == ord('r'):
            set_trackbars_from_params(default_params)
            print("[i] Parametreler sıfırlandı.")
        elif key == ord('s'):
            save_profile(params, view_mode)
        elif key == ord('l'):
            p, v = load_profile()
            if p:
                set_trackbars_from_params(p)
                view_mode = v
        elif key == ord('t'):
            if has_tracker:
                has_tracker = False
                tracker = None
                bbox = None
                print("[i] Takip kapatıldı.")
            else:
                freeze = frame.copy()
                roi = cv2.selectROI(
                    "View", freeze, fromCenter=False, showCrosshair=True)
                if roi and all(v > 0 for v in roi[2:]):
                    tracker = make_tracker()
                    if tracker is None:
                        print(
                            "[!] Tracker olusturulamadi. (opencv-contrib kurulu mu?)")
                    else:
                        ok = tracker.init(frame, roi)
                        has_tracker = ok
                        bbox = roi if ok else None
                        print(
                            "[i] Takip basladi." if ok else "[!] Takip baslatilamadi.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
