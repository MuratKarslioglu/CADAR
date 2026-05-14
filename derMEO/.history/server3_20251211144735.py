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

# Log kirliliğini önle
logging.getLogger("aiohttp").setLevel(logging.WARNING)
logging.getLogger("aiortc").setLevel(logging.WARNING)

# ===========================
# 2. GÜÇLENDİRİLMİŞ VIDEO KAYDEDİCİ (.mp4)
# ===========================
class AsyncVideoWriter:
    def __init__(self, filepath):
        self.filepath = filepath
        self.queue = queue.Queue()
        self.running = True
        self.writer = None 
        self.width = 0
        self.height = 0
        
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def write_frame(self, frame_bgr):
        if self.running:
            self.queue.put(frame_bgr)

    def stop(self):
        self.running = False
        self.thread.join()
        if self.writer:
            self.writer.release()

    def _worker(self):
        while True:
            try:
                frame = self.queue.get(timeout=1)
            except queue.Empty:
                if not self.running:
                    break
                continue
            
            # Frame boyutlarını al
            h, w = frame.shape[:2]

            # --- INIT KISMI ---
            if self.writer is None:
                self.width = w
                self.height = h
                # Windows için daha kararlı olan mp4v codec'ini kullanıyoruz
                fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
                self.writer = cv2.VideoWriter(self.filepath, fourcc, 20.0, (w, h))
            
            # --- HATA ÖNLEYİCİ (RESIZE) ---
            # Eğer gelen frame, writer'ın başladığı boyuttan farklıysa (örn: telefon döndü)
            # Frame'i resize et ki 'Failed to write' hatası alıp video bozulmasın.
            if self.writer is not None:
                if (w != self.width) or (h != self.height):
                    frame = cv2.resize(frame, (self.width, self.height))
                
                self.writer.write(frame)
            
            self.queue.task_done()

# ===========================
# 3. TRACK YÖNETİMİ
# ===========================
class DeviceTrack:
    def __init__(self, track, client_id):
        self.track = track
        self.client_id = client_id # Ham ID (Örn: Cihaz #1)
        # Dosya sistemi için güvenli ID oluştur (Örn: Cihaz_1)
        self.safe_id = re.sub(r'[^a-zA-Z0-9_\-]', '_', client_id)
        
        self.recorder = None
        self.task = asyncio.create_task(self.consume_frames()) 

    async def consume_frames(self):
        while True:
            try:
                frame = await self.track.recv()
                
                if self.recorder and self.recorder.running:
                    img = frame.to_ndarray(format="bgr24")
                    self.recorder.write_frame(img)
                    
            except Exception:
                break

    def start_recording(self):
        if self.recorder and self.recorder.running:
            return 
        
        # MP4 uzantılı dosya adı
        filename = f"{self.safe_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        path = os.path.join(VIDEO_DIR, filename)
        
        self.recorder = AsyncVideoWriter(path)
        print(f"   └─ 🎥 Kayıt Başladı: {filename}")

    def stop_recording(self):
        if self.recorder:
            self.recorder.stop()
            self.recorder = None
            print(f"   └─ ⏹️ Kayıt Durdu: {self.client_id}")

# ===========================
# 4. SERVER MİMARİSİ
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
                print(f"✅ Cihaz Bağlandı: {raw_id}")
                local_track = DeviceTrack(track, raw_id)
                self.device_tracks[raw_id] = local_track

        @pc.on("connectionstatechange")
        async def on_connection_change():
            if pc.connectionState in ["failed", "closed"]:
                print(f"❌ Cihaz Koptu: {raw_id}")
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

    # --- MAIN PC ---

    async def start_record(self, request):
        print("💻 Main PC: Kayıt Başlatma Emri.")
        count = 0
        for track in self.device_tracks.values():
            track.start_recording()
            count += 1
        return web.json_response({"status": "started", "device_count": count})

    async def stop_record(self, request):
        print("💻 Main PC: Kayıt Durdurma Emri.")
        for track in self.device_tracks.values():
            track.stop_recording()
        return web.json_response({"status": "stopped"})

    async def list_videos(self, request):
        print("💻 Main PC: Liste İstedi.")
        files = os.listdir(VIDEO_DIR)
        return web.json_response(files)

    async def download_video(self, request):
        # Dosya adını güvenli şekilde al
        filename = request.match_info.get('name') 
        if not filename:
            filename = request.rel_url.query.get("file")
            
        if not filename:
             return web.Response(status=400, text="Dosya adı yok.")

        # URL Decode işlemi (Gerekirse) ve Basename güvenliği
        filename = os.path.basename(filename)
        path = os.path.join(VIDEO_DIR, filename)

        print(f"💻 İndirme İsteği: {filename}")
        
        if os.path.exists(path):
            return web.FileResponse(path)
        
        print(f"❌ Dosya Yok: {path}")
        return web.Response(status=404, text="File not found")

# ===========================
# 5. RUN
# ===========================
def main():
    server = MediaServer()
    app = web.Application()

    app.router.add_post("/offer", server.handle_offer)
    app.router.add_post("/candidate", server.handle_candidate)
    
    app.router.add_post("/start_record", server.start_record)
    app.router.add_post("/stop_record", server.stop_record)
    app.router.add_get("/list_videos", server.list_videos)
    
    app.router.add_get("/download_video", server.download_video)
    app.router.add_get("/download_video/{name}", server.download_video)

    print("🚀 SERVER HAZIR (Safe Mode) - http://0.0.0.0:8080")
    print("📂 Kayıtlar .mp4 formatında 'videos' klasörüne yapılacak.")
    
    web.run_app(app, host="0.0.0.0", port=8080, access_log=None)

if __name__ == "__main__":
    main()