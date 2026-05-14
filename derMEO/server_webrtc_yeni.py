import json
import uuid
import os
from datetime import datetime

import cv2
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack

# ===========================
#   GLOBALS
# ===========================
VIDEO_DIR = "videos"
os.makedirs(VIDEO_DIR, exist_ok=True)

pcs = set()
android_tracks = {}          # {client_id: track}
viewer_pcs = set()

device_recorders = {}        # {client_id: {"writer":..., "filepath":...}}


# ===========================
#   FORWARD TRACK
# ===========================
class ForwardTrack(VideoStreamTrack):
    def __init__(self, src, client_id):
        super().__init__()
        self.src = src
        self.client_id = client_id

    async def recv(self):
        frame = await self.src.recv()

        # Convert WebRTC → OpenCV (BGR)
        img = frame.to_ndarray(format="rgb24")
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # DEVICE KAYIT AKTIF MI?
        if self.client_id in device_recorders:
            rec = device_recorders[self.client_id]
            writer = rec["writer"]

            # Eğer writer ilk frame’de açılmadıysa açalım
            if writer is None:
                h, w, _ = img.shape
                filepath = rec["filepath"]

                fourcc = cv2.VideoWriter_fourcc(*"XVID")
                wtr = cv2.VideoWriter(filepath, fourcc, 20.0, (w, h))
                device_recorders[self.client_id]["writer"] = wtr

            # Frame'i yaz
            wtr = device_recorders[self.client_id]["writer"]
            if wtr is not None:
                wtr.write(img)

        return frame


# ===========================
#   OFFER HANDLE
# ===========================
async def handle_offer(request):
    params = await request.json()
    client_id = params.get("client_id", str(uuid.uuid4()))
    is_android = params.get("is_android", False)

    print(f"\n📡 OFFER from {client_id}  Android={is_android}")

    offer = RTCSessionDescription(params["sdp"], params["type"])
    pc = RTCPeerConnection()
    pcs.add(pc)

    if is_android:

        @pc.on("track")
        async def on_track(track):
            if track.kind == "video":
                print(f"🔥 Android video connected → {client_id}")
                android_tracks[client_id] = track

                # Bu cihazı tüm viewer’lara ilet
                for viewer in viewer_pcs:
                    viewer.addTrack(ForwardTrack(track, client_id))

    else:
        # PC VIEWER
        print("🖥 PC Viewer bağlandı")
        viewer_pcs.add(pc)

        # Viewer bağlanınca tüm cihazları ona gönder
        for dev_id, track in android_tracks.items():
            pc.addTrack(ForwardTrack(track, dev_id))

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.json_response({
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type
    })


# ===========================
#   RECORDING API
# ===========================
async def start_record(request):
    data = await request.json()
    dev_id = data.get("client_id")

    if dev_id not in android_tracks:
        return web.json_response({"error": "device_not_connected"})

    filename = f"{dev_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.avi"
    filepath = os.path.join(VIDEO_DIR, filename)

    print(f"🔴 START RECORD for {dev_id} → {filepath}")

    device_recorders[dev_id] = {"writer": None, "filepath": filepath}

    return web.json_response({"status": "started", "file": filename})


async def stop_record(request):
    data = await request.json()
    dev_id = data.get("client_id")

    if dev_id in device_recorders:
        rec = device_recorders[dev_id]
        writer = rec["writer"]
        if writer is not None:
            print(f"💾 Writer closed → {dev_id}")
            writer.release()

        del device_recorders[dev_id]

    return web.json_response({"status": "stopped"})


async def list_videos(request):
    return web.json_response(os.listdir(VIDEO_DIR))


async def download_video(request):
    name = request.rel_url.query.get("file")
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


# ===========================
#   MAIN SERVER
# ===========================
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
