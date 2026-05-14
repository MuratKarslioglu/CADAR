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

# Log kirliliğini engelle
logging.getLogger("aiohttp").setLevel(logging.WARNING)
logging.getLogger("aiortc").setLevel(logging.WARNING)

# ===========================
# 2. GÜÇLENDİRİLMİŞ MP4 KAYDEDİCİ
# ===========================
class AsyncVideoWriter:
    def __init__(self, filepath):
        self.filepath = filepath
        self.queue = queue.Queue()
        self.running = True
        self.writer = None 
        self.frame_count = 0
        
        # Thread başlat
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
            print(f"\n✅ Dosya Kaydedildi: {self.filepath} | Toplam Kare: {self.frame_count}")

    def _worker(self):
        while True:
            try:
                frame = self.queue.get(timeout=1)
            except queue.Empty:
                if not self.running:
                    break
                continue
            
            # Gelen karenin orijinal boyutları
            h, w = frame.shape[:2]

            # --- INIT (İLK KAREDE YAZICIYI KUR) ---
            if self.writer is None:
                # KRİTİK: MP4 codec'leri tek sayılardan (örn: 1079) nefret eder.
                # Genişlik ve yüksekliği en yakın çift sayıya yuvarlıyoruz.
                safe_w = w if w % 2 == 0 else w - 1
                safe_h = h if h % 2 == 0 else h - 1
                
                # Windows ve OpenCV için en uyumlu MP4 codec'i: 'mp4v'
                fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
                self.writer = cv2.VideoWriter(self.filepath, fourcc, 20.0, (safe_w, safe_h))
                print(f"🎬 MP4 Yazıcı Başladı ({safe_w}x{safe_h}): {self.filepath}")

            # --- YAZMA İŞLEMİ ---
            if self.writer is not None:
                # Hedef boyutları al
                target_w = int(self.writer.get(cv2.CAP_PROP_FRAME_WIDTH))
                target_h = int(self.writer.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                # Eğer gelen kare, hedef boyuta uymuyorsa resize et (Çökme önleyici)
                if (w != target_w) or (h != target_h):
                    frame = cv2.resize(frame, (target_w, target_h))
                
                self.writer.write(frame)
                self.frame_count += 1
                
                # Çalıştığını görmek için ekrana nokta bas
                if self.frame_count % 30 == 0:
                    print(".", end="", flush=True)
            
            self.queue.task_done()

# ===========================
# 3. STREAM TRACK
# ===========================
class DeviceTrack:
    def __init__(self, track, client_id):
        self.track = track
        self.client_id = client_id
        # Dosya adı güvenliği (boşlukları _ yap)
        self.safe_id = re.sub(r'[^a-zA-Z0-9_\-]', '_', client_id)
        
        self.recorder = None
        self.task = asyncio.create_task(self.consume_frames()) 

    async def consume_frames(self):
        while True:
            try:
                frame = await self.track.recv()
                
                # Sadece kayıt aktifken writer'a gönder
                if self.recorder and self.recorder.running:
                    try:
                        img = frame.to_ndarray(format="bgr24")
                        self.recorder.write_frame(img)
                    except Exception:
                        pass # Frame hatası olursa akışı bozma

            except Exception:
                break

    def start_recording(self):
        if self.recorder and self.recorder.running:
            return 
        
        # ARTIK .MP4 FORMATINDAYIZ
        filename = f"{self.safe_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        path = os.path.join(VIDEO_DIR, filename)
        
        self.recorder = AsyncVideoWriter(path)
        print(f"\n   └─ 🎥 Kayıt Emri: {filename}")

    def stop_recording(self):
        if self.recorder:
            self.recorder.stop()
            self.recorder = None

# ===========================
# 4. SERVER YÖNETİMİ
# ===========================
class MediaServer:
    def __init__(self):
        self.pcs = set()
        self.device_tracks = {} 

    # --- WEBRTC ---
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

    # --- MAIN PC ---
    async def handle_main_pc_connect(self, request):
        print("\n" + "="*40)
        print("💻 MAIN PC SİSTEME GİRİŞ YAPTI")
        print("="*40 + "\n")
        return web.Response(text="Bağlantı Başarılı")

    async def start_record(self, request):
        print("\n💻 START: Tüm Cihazlar Kaydediliyor...")
        count = 0
        if not self.device_tracks:
             print("⚠️ UYARI: Cihaz yok!")
             return web.json_response({"status": "failed", "reason": "no_devices"})

        for track in self.device_tracks.values():
            track.start_recording()
            count += 1
        return web.json_response({"status": "started", "device_count": count})

    async def stop_record(self, request):
        print("\n💻 STOP: Kayıtlar Kapatılıyor...")
        for track in self.device_tracks.values():
            track.stop_recording()
        return web.json_response({"status": "stopped"})

    async def list_videos(self, request):
        files = os.listdir(VIDEO_DIR)
        return web.json_response(files)

    async def download_video(self, request):
        name = request.match_info.get('name', request.rel_url.query.get("file"))
        if not name: return web.Response(status=400, text="Eksik dosya adı")
        
        name = os.path.basename(name)
        path = os.path.join(VIDEO_DIR, name)
        
        print(f"💻 İndiriliyor: {name}")
        if os.path.exists(path):
            return web.FileResponse(path)
        
        print(f"❌ Bulunamadı: {path}")
        return web.Response(status=404, text="Dosya yok")

# ===========================
# 5. BAŞLAT
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

    print("\n🚀 SERVER HAZIR (MP4 MODU) - http://0.0.0.0:8080")
    print("---------------------------------------------------")
    web.run_app(app, host="0.0.0.0", port=8080, access_log=None)

if __name__ == "__main__":
    main()