import asyncio
import json
import os
import uuid
import threading
import queue
import cv2
import logging
from datetime import datetime
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack

# ===========================
# 1. AYARLAR & TEMİZLİK
# ===========================
VIDEO_DIR = "videos"
os.makedirs(VIDEO_DIR, exist_ok=True)

# Gereksiz kütüphane loglarını sustur
logging.getLogger("aiohttp").setLevel(logging.WARNING)
logging.getLogger("aiortc").setLevel(logging.WARNING)

# ===========================
# 2. AKILLI VIDEO KAYDEDİCİ (Threaded)
# ===========================
class AsyncVideoWriter:
    def __init__(self, filepath):
        self.filepath = filepath
        self.queue = queue.Queue()
        self.running = True
        self.writer = None # İlk frame gelene kadar oluşturmuyoruz
        
        # Yazma işlemini arka plan thread'ine atıyoruz
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
            
            # --- KRİTİK DÜZELTME ---
            # Writer'ı ilk gelen karenin boyutuna göre oluşturuyoruz.
            # Böylece 1080p gelirse 1080p, 720p gelirse 720p kaydeder.
            if self.writer is None:
                h, w = frame.shape[:2]
                # Windows uyumluluğu için XVID, duruma göre mp4v de denenebilir
                fourcc = cv2.VideoWriter_fourcc(*'XVID') 
                self.writer = cv2.VideoWriter(self.filepath, fourcc, 20.0, (w, h))
            
            self.writer.write(frame)
            self.queue.task_done()

# ===========================
# 3. TRACK YÖNETİCİSİ
# ===========================
class DeviceTrack:
    """
    WebRTC Track'ini dinler ve frames'leri yönetir.
    Video kaydı isteği gelirse AsyncVideoWriter'a yönlendirir.
    """
    def __init__(self, track, client_id):
        self.track = track
        self.client_id = client_id
        self.recorder = None
        self.task = asyncio.create_task(self.consume_frames()) # Arka planda çalışmaya başla

    async def consume_frames(self):
        while True:
            try:
                frame = await self.track.recv()
                
                # Eğer kayıtçı aktifse frame'i gönder
                if self.recorder and self.recorder.running:
                    # WebRTC Frame -> BGR formatına çevir (OpenCV için)
                    img = frame.to_ndarray(format="bgr24")
                    self.recorder.write_frame(img)
                    
            except Exception:
                # Bağlantı koptuğunda döngü kırılır
                break

    def start_recording(self):
        if self.recorder and self.recorder.running:
            return 
        
        filename = f"{self.client_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.avi"
        path = os.path.join(VIDEO_DIR, filename)
        
        # Boyut vermiyoruz, ilk frame'den otomatik algılayacak
        self.recorder = AsyncVideoWriter(path)
        print(f"🔴 KAYIT BAŞLADI: {self.client_id} -> {filename}")

    def stop_recording(self):
        if self.recorder:
            self.recorder.stop()
            self.recorder = None
            print(f"✅ KAYIT BİTTİ: {self.client_id}")

# ===========================
# 4. SERVER MİMARİSİ
# ===========================
class MediaServer:
    def __init__(self):
        self.pcs = set()
        self.device_tracks = {} # {client_id: DeviceTrack}

    async def handle_offer(self, request):
        params = await request.json()
        client_id = params.get("client_id", str(uuid.uuid4()))
        
        offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
        pc = RTCPeerConnection()
        self.pcs.add(pc)

        @pc.on("track")
        def on_track(track):
            if track.kind == "video":
                print(f"📱 Cihaz Bağlandı: {client_id}")
                # Track'i sarmalayıp saklıyoruz
                local_track = DeviceTrack(track, client_id)
                self.device_tracks[client_id] = local_track

        @pc.on("connectionstatechange")
        async def on_connection_change():
            if pc.connectionState in ["failed", "closed"]:
                print(f"❌ Cihaz Koptu: {client_id}")
                await self.cleanup(pc, client_id)

        await pc.setRemoteDescription(offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        return web.json_response({
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type
        })

    async def handle_candidate(self, request):
        return web.Response(status=200) # Trickle ICE hatasını önlemek için

    async def cleanup(self, pc, client_id):
        self.pcs.discard(pc)
        if client_id in self.device_tracks:
            self.device_tracks[client_id].stop_recording()
            del self.device_tracks[client_id]
        await pc.close()

    # --- KONTROL ENDPOINTLERİ ---

    async def start_record(self, request):
        """Tüm bağlı cihazlarda kaydı başlatır."""
        count = 0
        for track in self.device_tracks.values():
            track.start_recording()
            count += 1
        
        if count == 0:
            print("⚠️ Kayıt başlatılacak cihaz yok.")
        
        return web.json_response({"status": "started", "device_count": count})

    async def stop_record(self, request):
        """Tüm kayıtları durdurur."""
        for track in self.device_tracks.values():
            track.stop_recording()
        return web.json_response({"status": "stopped"})

    async def list_videos(self, request):
        files = os.listdir(VIDEO_DIR)
        return web.json_response(files)

    async def download_video(self, request):
        name = request.match_info.get('name', request.rel_url.query.get("file"))
        path = os.path.join(VIDEO_DIR, name)
        if os.path.exists(path):
            return web.FileResponse(path)
        return web.Response(status=404, text="File not found")

# ===========================
# 5. UYGULAMAYI BAŞLAT
# ===========================
def main():
    server = MediaServer()
    app = web.Application()

    # Endpointler
    app.router.add_post("/offer", server.handle_offer)
    app.router.add_post("/candidate", server.handle_candidate)
    
    app.router.add_post("/start_record", server.start_record)
    app.router.add_post("/stop_record", server.stop_record)
    
    app.router.add_get("/list_videos", server.list_videos)
    app.router.add_get("/download_video", server.download_video)
    app.router.add_get("/download_video/{name}", server.download_video)

    print("🚀 Server Hazır (Sessiz Mod) - http://0.0.0.0:8080")
    print("---------------------------------------------------")
    web.run_app(app, host="0.0.0.0", port=8080, access_log=None) # Access log kapatıldı

if __name__ == "__main__":
    main()