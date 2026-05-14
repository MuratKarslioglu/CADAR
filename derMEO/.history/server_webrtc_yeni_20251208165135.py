import json
import uuid
import os
from datetime import datetime

import cv2
import numpy as np
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceCandidate, VideoStreamTrack

# ============================
#   GLOBALS
# ============================
VIDEO_DIR = "videos"
os.makedirs(VIDEO_DIR, exist_ok=True)

pcs = set()
android_tracks = {}
viewer_pcs = set()

recording = False
record_writer = None
record_filepath = None


# ============================
#   FORWARD TRACK
# ============================
class ForwardTrack(VideoStreamTrack):
    def __init__(self, src, client_id):
        super().__init__()
        self.src = src
        self.client_id = client_id

    async def recv(self):
        global recording, record_writer, record_filepath

        frame = await self.src.recv()

        # WebRTC frame'ini OpenCV için BGR'e çevir
        img = frame.to_ndarray(format="rgb24")
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # İlk frame'de writer aç
        if recording and record_writer is None:
            h, w, _ = img.shape
            print(f"🎞 VideoWriter açılıyor → {w}x{h} path={record_filepath}")

            # 🔧 Codec + container: XVID + AVI
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            writer = cv2.VideoWriter(
                record_filepath,
                fourcc,
                20.0,
                (w, h)
            )

            # Açılabildi mi?
            if not writer.isOpened():
                print("❌ VideoWriter AÇILAMADI!")
            else:
                print("✅ VideoWriter başarıyla açıldı.")

            record_writer = writer

        # Frame yaz
        if recording and record_writer is not None and record_writer.isOpened():
            record_writer.write(img)
        else:
            if recording:
                # Sadece debug için, sürekli spam olmasın diye istersen yorum satırı yaparsın
                print("⚠ recording=True ama writer açık değil.")

        return frame



# ============================
#  OFFER HANDLE
# ============================
async def handle_offer(request):
    params = await request.json()
    client_id = params.get("client_id", str(uuid.uuid4()))
    is_android = params.get("is_android", False)

    print(f"\n📡 OFFER from {client_id}  Android={is_android}")

    offer = RTCSessionDescription(params["sdp"], params["type"])
    pc = RTCPeerConnection()
    pcs.add(pc)

    # ---- ANDROID ----
    if is_android:

        @pc.on("track")
        async def on_track(track):
            if track.kind == "video":
                print(f"🔥 Android video track connected → {client_id}")
                android_tracks[client_id] = track

                # Tüm PC viewerlara ilet
                for viewer in viewer_pcs:
                    viewer.addTrack(ForwardTrack(track, client_id))
    
    else:
        # ---- PC VIEWER ----
        print("🖥 PC Viewer bağlandı")
        viewer_pcs.add(pc)

        # Var olan android kameralarını yeni bağlanan PC’ye ilet
        for dev_id, track in android_tracks.items():
            pc.addTrack(ForwardTrack(track, dev_id))

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.json_response({
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type
    })


# ============================
#   RECORDING API
# ============================
async def start_record(request):
    global recording, record_writer, record_filepath

    data = await request.json()
    dev_id = data.get("client_id", "device")

    # 🔴 UZANTIYI .avi YAPTIK
    filename = f"{dev_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.avi"
    record_filepath = os.path.join(VIDEO_DIR, filename)

    print(f"🔴 START RECORD → {record_filepath}")

    recording = True
    record_writer = None  # writer ilk frame’de açılacak

    return web.json_response({"status": "started", "file": filename})



async def stop_record(request):
    global recording, record_writer

    print("🟢 STOP RECORD")
    recording = False

    if record_writer is not None:
        print("💾 Writer kapatılıyor...")
        record_writer.release()
        record_writer = None

    return web.json_response({"status": "stopped"})



async def list_videos(request):
    return web.json_response(os.listdir(VIDEO_DIR))


async def download_video(request):
    name = request.rel_url.query.get("file")
    if not name:
        return web.Response(text="NO FILE", status=400)

    path = os.path.join(VIDEO_DIR, name)
    if not os.path.exists(path):
        return web.Response(text="NOT FOUND", status=404)

    return web.FileResponse(path)


async def delete_video(request):
    data = await request.json()
    name = data.get("file")

    path = os.path.join(VIDEO_DIR, name)
    if os.path.exists(path):
        os.remove(path)
        return web.Response(text="OK")

    return web.Response(text="FILE NOT FOUND", status=404)


# ============================
#   MAIN SERVER
# ============================
def main():
    app = web.Application()
    app.router.add_post("/offer", handle_offer)
    app.router.add_post("/start_record", start_record)
    app.router.add_post("/stop_record", stop_record)
    app.router.add_get("/list_videos", list_videos)
    app.router.add_get("/download_video", download_video)
    app.router.add_post("/delete_video", delete_video)

    print("🚀 Server running at http://192.168.1.2:8080")
    web.run_app(app, host="192.168.1.2", port=8080)


if __name__ == "__main__":
    main()
