import json
import uuid
import os
from datetime import datetime

import cv2
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceCandidate, VideoStreamTrack
from aiortc.contrib.media import MediaRelay


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

relay = MediaRelay()   # 🔥 Çoklu cihaz için zorunlu!


# ============================
#   FORWARD TRACK (Kaydetmek için)
# ============================
class ForwardTrack(VideoStreamTrack):
    def __init__(self, src, client_id):
        super().__init__()
        self.src = src
        self.client_id = client_id

    async def recv(self):
        global recording, record_writer, record_filepath

        frame = await self.src.recv()

        # OpenCV BGR formatına çevir
        img = frame.to_ndarray(format="rgb24")
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Writer ilk frame’de oluşturulacak
        if recording and record_writer is None:
            h, w, _ = img.shape
            print(f"🎞 VideoWriter açılıyor → {w}x{h} path={record_filepath}")

            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            writer = cv2.VideoWriter(record_filepath, fourcc, 20.0, (w, h))

            if not writer.isOpened():
                print("❌ VideoWriter açılamadı!")
            else:
                print("✅ VideoWriter başarıyla açıldı.")

            record_writer = writer

        # Kayıt aktifse frame yaz
        if recording and record_writer is not None and record_writer.isOpened():
            record_writer.write(img)

        return frame


# ============================
#  OFFER HANDLE
# ============================
async def handle_candidate(request):
    params = await request.json()
    client_id = params.get("client_id")
    candidate_data = params.get("candidate")

    print(f"📨 ICE candidate alındı → {client_id}")

    for pc in pcs:
        try:
            candidate = RTCIceCandidate(
                candidate=candidate_data["candidate"],
                sdpMid=candidate_data["sdpMid"],
                sdpMLineIndex=candidate_data["sdpMLineIndex"]
            )
            await pc.addIceCandidate(candidate)
        except Exception as e:
            print("❌ ICE ekleme hatası:", e)

    return web.json_response({"status": "ok"})

async def handle_offer(request):
    params = await request.json()
    client_id = params.get("client_id", str(uuid.uuid4()))
    is_android = params.get("is_android", False)

    print(f"\n📡 OFFER from {client_id}  Android={is_android}")

    offer = RTCSessionDescription(params["sdp"], params["type"])
    pc = RTCPeerConnection()
    pcs.add(pc)

    # ---- ANDROID CIHAZ ----
    if is_android:

        @pc.on("track")
        async def on_track(track):
            if track.kind == "video":
                print(f"🔥 Android video track geldi → {client_id}")
                android_tracks[client_id] = track

                # Viewer'lara bu track'i gönder
                for viewer in viewer_pcs:
                    print(f"➡️ Track viewer'a gönderiliyor → {client_id}")
                    viewer.addTrack(relay.subscribe(track))  # 🔥 Çok önemli

    # ---- PC VIEWER ----
    else:
        print("🖥 PC Viewer bağlandı")
        viewer_pcs.add(pc)

        # Mevcut tüm kameraları yeni viewer'a ilet
        for dev_id, track in android_tracks.items():
            print(f"➡️ Viewer'a eski cihaz stream'i gönderiliyor → {dev_id}")
            pc.addTrack(relay.subscribe(track))  # 🔥 Çok önemli

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

    filename = f"{dev_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.avi"
    record_filepath = os.path.join(VIDEO_DIR, filename)

    print(f"🔴 START RECORD → {record_filepath}")

    recording = True
    record_writer = None

    return web.json_response({"status": "started", "file": filename})


async def stop_record(request):
    global recording, record_writer

    print("🟢 STOP RECORD")
    recording = False

    if record_writer is not None:
        record_writer.release()
        record_writer = None
        print("💾 Writer kapatıldı")

    return web.json_response({"status": "stopped"})


# ============================
#   VIDEO FILE HANDLERS
# ============================
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
    app.router.add_post("/offer", handle_offer)
    app.router.add_post("/candidate", handle_candidate)


    print("🚀 Server running at http://192.168.1.2:8080")
    web.run_app(app, host="192.168.1.2", port=8080)


if __name__ == "__main__":
    main()
