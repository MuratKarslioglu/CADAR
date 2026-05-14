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
# 1. AYARLAR & LOGLAMA
# ===========================
VIDEO_DIR = "videos"
os.makedirs(VIDEO_DIR, exist_ok=True)

# Kütüphane kalabalığını sustur
logging.getLogger("aiohttp").setLevel(logging.WARNING)
logging.getLogger("aiortc").setLevel(logging.WARNING)

# ===========================
# 2. BAĞIMSIZ VIDEO KAYDEDİCİ (HER CİHAZ İÇİN AYRI)
# ===========================
class AsyncVideoWriter:
    def __init__(self, filepath):
        self.filepath = filepath
        self.queue = queue.Queue()
        self.running = True
        self.writer = None 
        
        # Her kayıt işlemi kendi Thread'inde çalışır (Performans için şart)
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
            
            # Dinamik Çözünürlük: İlk gelen frame neyse video o boyutta olur
            if self.writer is None:
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'XVID') 
                self.writer = cv2.VideoWriter(self.filepath, fourcc, 20.0, (w, h))
            
            self.writer.write(frame)
            self.queue.task_done()

# ===========================
# 3. CİHAZ YÖNETİCİSİ (TRACK)
# ===========================
class DeviceTrack:
    """
    Android cihazdan gelen görüntüyü karşılar.
    Kayıt emri gelene kadar görüntüyü boşa akıtır (CPU şişmez).
    """
    def __init__(self, track, client_id):
        self.track = track
        self.client_id = client_id
        self.recorder = None
        # Görüntü akışını başlatan asenkron görev
        self.task = asyncio.create_task(self.consume_frames()) 

    async def consume_frames(self):
        while True:
            try:
                frame = await self.track.recv()
                
                # SADECE KAYIT AKTİFSE diske gönderir
                if self.recorder and self.recorder.running:
                    img = frame.to_ndarray(format="bgr24")
                    self.recorder.write_frame(img)
                    
            except Exception:
                # Bağlantı koparsa döngü biter
                break

    def start_recording(self):
        if self.recorder and self.recorder.running:
            return # Zaten kayıt yapıyor
        
        # Dosya ismi: CihazID_TarihSaat.avi
        filename = f"{self.client_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.avi"
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
        self.device_tracks = {} # {client_id: DeviceTrack}

    # --- ANDROID BAĞLANTI HANDLE ---
    async def handle_offer(self, request):
        params = await request.json()
        client_id = params.get("client_id", str(uuid.uuid4()))
        
        # Log: Cihaz bağlanmaya çalışıyor
        # print(f"📡 Bağlantı İsteği: {client_id}") 

        offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
        pc = RTCPeerConnection()
        self.pcs.add(pc)

        @pc.on("track")
        def on_track(track):
            if track.kind == "video":
                print(f"✅ Cihaz Bağlandı ve Görüntü Hazır: {client_id}")
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

    async def cleanup(self, pc, client_id):
        self.pcs.discard(pc)
        if client_id in self.device_tracks:
            self.device_tracks[client_id].stop_recording()
            del self.device_tracks[client_id]
        await pc.close()

    async def handle_candidate(self, request):
        return web.Response(status=200)

    # --- MAIN PC (KONTROL PANELİ) ---

    async def start_record(self, request):
        print("💻 Main PC: Kayıt Başlatma Emri Gönderdi.")
        count = 0
        for track in self.device_tracks.values():
            track.start_recording()
            count += 1
        
        if count == 0:
            print("⚠️ UYARI: Hiçbir cihaz bağlı değil, kayıt başlamadı.")
            return web.json_response({"status": "failed", "reason": "no_devices"})
            
        return web.json_response({"status": "started", "device_count": count})

    async def stop_record(self, request):
        print("💻 Main PC: Kayıt Durdurma Emri Gönderdi.")
        for track in self.device_tracks.values():
            track.stop_recording()
        return web.json_response({"status": "stopped"})

    async def list_videos(self, request):
        print("💻 Main PC: Video Listesini İstedi.")
        try:
            files = os.listdir(VIDEO_DIR)
            return web.json_response(files)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    # --- İNDİRME İŞLEMİ (DÜZELTİLDİ) ---
    async def download_video(self, request):
        # Dosya adını hem path parametresinden hem query'den alabilir
        filename = request.match_info.get('name') 
        if not filename:
            filename = request.rel_url.query.get("file")
            
        if not filename:
             return web.Response(status=400, text="Dosya adı belirtilmedi.")

        # Güvenlik ve Yol Düzeltme
        filename = os.path.basename(filename) # ../ gibi saldırıları engeller
        path = os.path.join(VIDEO_DIR, filename)

        print(f"💻 Main PC: İndirme İsteği -> {filename}")
        
        if os.path.exists(path):
            return web.FileResponse(path)
        
        print(f"❌ Dosya Bulunamadı: {path} (Aranan yol buydu)")
        return web.Response(status=404, text=f"File not found on server: {filename}")

# ===========================
# 5. SERVER BAŞLATMA
# ===========================
def main():
    server = MediaServer()
    app = web.Application()

    # Android Endpointleri
    app.router.add_post("/offer", server.handle_offer)
    app.router.add_post("/candidate", server.handle_candidate)
    
    # Main PC Endpointleri
    app.router.add_post("/start_record", server.start_record)
    app.router.add_post("/stop_record", server.stop_record)
    app.router.add_get("/list_videos", server.list_videos)
    
    # İndirme Endpointleri (İki türlü de yakalar)
    app.router.add_get("/download_video", server.download_video)
    app.router.add_get("/download_video/{name}", server.download_video)

    print("---------------------------------------------------")
    print("🚀 SERVER HAZIR - http://0.0.0.0:8080")
    print("📂 Kayıt Klasörü: /videos")
    print("---------------------------------------------------")
    
    # Access log'u kapattım, sadece kendi printlerimiz görünecek
    web.run_app(app, host="0.0.0.0", port=8080, access_log=None)

if __name__ == "__main__":
    main()