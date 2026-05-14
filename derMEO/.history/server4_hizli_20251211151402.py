import asyncio
import json
import os
import uuid
import threading
import queue
import re
import cv2
import logging
from datetime import datetime
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack

# ===========================
# 1. AYARLAR
# ===========================
VIDEO_DIR = "videos"
os.makedirs(VIDEO_DIR, exist_ok=True)

# Logları sustur
logging.getLogger("aiohttp").setLevel(logging.WARNING)
logging.getLogger("aiortc").setLevel(logging.WARNING)

# ===========================
# 2. TURBO MP4 KAYDEDİCİ (FPS FIX + DROP FRAME)
# ===========================
class AsyncVideoWriter:
    def __init__(self, filepath):
        self.filepath = filepath
        # MAXSIZE: Kuyrukta en fazla 10 kare biriksin. 
        # Daha fazla gelirse CPU yetişemiyor demektir, eskileri atıp günceli alacağız.
        self.queue = queue.Queue(maxsize=10) 
        self.running = True
        self.writer = None 
        self.frame_count = 0
        self.dropped_frames = 0
        self.target_width = 0
        self.target_height = 0
        
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def write_frame(self, frame_bgr):
        if not self.running:
            return

        try:
            # put_nowait: Eğer kuyruk doluysa bekleme yapmaz, hata fırlatır (Full)
            self.queue.put_nowait(frame_bgr)
        except queue.Full:
            # Kuyruk dolu! Demek ki diske yazma hızımız, gelen hızdan yavaş.
            # Videonun geriden gelmemesi için en eski kareyi çöpe at, yeniyi ekle.
            try:
                self.queue.get_nowait() # Eskiyi at
                self.queue.put_nowait(frame_bgr) # Yeniyi koy
                self.dropped_frames += 1
            except:
                pass

    def stop(self):
        self.running = False
        self.thread.join()
        if self.writer:
            self.writer.release()
            print(f"\n✅ Kayıt Bitti: {self.filepath}")
            print(f"   📊 İstatistik: {self.frame_count} kare yazıldı | {self.dropped_frames} kare atlandı (Performans ayarı)")

    def _worker(self):
        while True:
            try:
                frame = self.queue.get(timeout=1)
            except queue.Empty:
                if not self.running:
                    break
                continue
            
            h, w = frame.shape[:2]

            # --- INIT ---
            if self.writer is None:
                # Çift sayı kuralı
                safe_w = w if w % 2 == 0 else w - 1
                safe_h = h if h % 2 == 0 else h - 1
                
                self.target_width = safe_w
                self.target_height = safe_h
                
                fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
                # HIZ AYARI: 20.0 yerine 30.0 yaptık. Android genelde 30 gönderir.
                self.writer = cv2.VideoWriter(self.filepath, fourcc, 30.0, (safe_w, safe_h))
                print(f"🎬 MP4 Başladı (30 FPS - {safe_w}x{safe_h}): {self.filepath}")

            # --- YAZMA ---
            if self.writer is not None:
                # Boyut değişimi kontrolü (Resize)
                if (w != self.target_width) or (h != self.target_height):
                    try:
                        frame = cv2.resize(frame, (self.target_width, self.target_height))
                    except:
                        continue
                
                self.writer.write(frame)
                self.frame_count += 1
                
                # Feedback
                if self.frame_count % 60 == 0: # Her 2 saniyede bir nokta
                    print(".", end="", flush=True)
            
            self.queue.task_done()

# ===========================
# 3. STREAM TRACK
# ===========================
class DeviceTrack:
    def __init__(self, track, client_id):
        self.track = track
        self.client_id = client_id
        self.safe_id = re.sub(r'[^a-zA-Z0-9_\-]', '_', client_id)
        
        self.recorder = None
        self.task = asyncio.create_task(self.consume_frames()) 

    async def consume_frames(self):
        while True:
            try:
                frame = await self.track.recv()
                
                if self.recorder and self.recorder.running:
                    try:
                        img = frame.to_ndarray(format="bgr24")
                        self.recorder.write_frame(img)
                    except Exception:
                        pass 
            except Exception:
                break

    def start_recording(self):
        if self.recorder and self.recorder.running:
            return 
        
        filename = f"{self.safe_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        path = os.path.join(VIDEO_DIR, filename)
        
        self.recorder = AsyncVideoWriter(path)
        print(f"\n   └─ 🎥 Kayıt Emri: {filename}")

    def stop_recording(self):
        if self.recorder:
            self.recorder.stop()
            self.recorder = None

# ===========================
# 4. SERVER CONTROLLER
# ===========================
class MediaServer:
    def __init__(self):
        self.pcs = set()
        self.device_tracks = {} 

    async def handle_offer(self, request):
        params = await request.json()
        raw_id = params.get("client_id", str(uuid.uuid4()))
        
        offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
        pc = RTCPeerConnection()
        self.pcs.add(pc)

        @pc.on("track")
        def on_track(track):
            if track.kind == "video":
                print(f"\n📱 Bağlandı: {raw_id}")
                local_track = DeviceTrack(track, raw_id)
                self.device_tracks[raw_id] = local_track

        @pc.on("connectionstatechange")
        async def on_connection_change():
            if pc.connectionState in ["failed", "closed"]:
                print(f"\n❌ Koptu: {raw_id}")
                await self.cleanup(pc, raw_id)

        await pc.setRemoteDescription(offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        return web.json_response({
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type
        })

    async def cleanup(self, pc, client_id):
        self.pcs.discard(pc)
        if client_id in self.device_tracks:
            self.device_tracks[client_id].stop_recording()
            del self.device_tracks[client_id]
        await pc.close()

    async def handle_candidate(self, request):
        return web.Response(status=200)

    async def handle_main_pc_connect(self, request):
        print("\n" + "="*40)
        print("💻 MAIN PC BAĞLANDI")
        print("="*40 + "\n")
        return web.Response(text="OK")

    async def start_record(self, request):
        print("\n💻 START: Kayıt Başlıyor...")
        count = 0
        if not self.device_tracks:
             print("⚠️ Cihaz yok!")
             return web.json_response({"status": "failed"})

        for track in self.device_tracks.values():
            track.start_recording()
            count += 1
        return web.json_response({"status": "started", "device_count": count})

    async def stop_record(self, request):
        print("\n💻 STOP: Kayıt Bitiyor...")
        for track in self.device_tracks.values():
            track.stop_recording()
        return web.json_response({"status": "stopped"})

    async def list_videos(self, request):
        files = os.listdir(VIDEO_DIR)
        return web.json_response(files)

    async def download_video(self, request):
        name = request.match_info.get('name', request.rel_url.query.get("file"))
        if not name: return web.Response(status=400, text="Eksik isim")
        
        name = os.path.basename(name)
        path = os.path.join(VIDEO_DIR, name)
        
        print(f"💻 İndiriliyor: {name}")
        if os.path.exists(path):
            return web.FileResponse(path)
        return web.Response(status=404, text="Yok")

# ===========================
# 5. RUN
# ===========================
def main():
    app = web.Application()
    server = MediaServer()

    app.router.add_post("/offer", server.handle_offer)
    app.router.add_post("/candidate", server.handle_candidate)
    app.router.add_get("/admin/connect", server.handle_main_pc_connect)
    app.router.add_post("/start_record", server.start_record)
    app.router.add_post("/stop_record", server.stop_record)
    app.router.add_get("/list_videos", server.list_videos)
    app.router.add_get("/download_video", server.download_video)
    app.router.add_get("/download_video/{name}", server.download_video)

    print("\n🚀 SERVER HAZIR (30 FPS TURBO) - http://0.0.0.0:8080")
    print("---------------------------------------------------")
    web.run_app(app, host="0.0.0.0", port=8080, access_log=None)

if __name__ == "__main__":
    main()