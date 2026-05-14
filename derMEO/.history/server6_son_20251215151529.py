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
# 1. AYARLAR & PERFORMANS
# ===========================
VIDEO_DIR = "videos"
os.makedirs(VIDEO_DIR, exist_ok=True)

logging.getLogger("aiohttp").setLevel(logging.WARNING)
logging.getLogger("aiortc").setLevel(logging.WARNING)

# AYNI ANDA KAÇ VİDEO DİSKE YAZILSIN? (İşlemci gücüne göre artırabilirsin)
WORKER_COUNT = 4 

# ===========================
# 2. PARALEL İŞÇİ HAVUZU (THREAD POOL)
# ===========================
SAVE_QUEUE = queue.Queue()

def save_worker(worker_id):
    """
    Her bir işçi bu fonksiyonu çalıştırır.
    Kuyruktan iş kapmaca oynarlar.
    """
    print(f"⚙️ İşçi #{worker_id} Hazır ve Bekliyor...")
    
    while True:
        # Kuyruktan işi al
        job = SAVE_QUEUE.get()
        
        filepath = job['filepath']
        frames = job['frames']
        target_w = job['width']
        target_h = job['height']
        fps = job['fps']
        
        print(f"\n⚡ [İşçi-{worker_id}] YAZIYOR: {filepath}")
        print(f"   📦 Kare Sayısı: {len(frames)} | Kalite: LANCZOS4 (En İyi)")

        try:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(filepath, fourcc, fps, (target_w, target_h))
            
            frame_counter = 0
            total_frames = len(frames)

            for frame in frames:
                # EN İYİ KALİTE İÇİN: INTER_LANCZOS4
                # Görüntü boyutunu bozmadan en keskin haliyle işler.
                if frame.shape[1] != target_w or frame.shape[0] != target_h:
                    try:
                        frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
                    except:
                        continue
                
                writer.write(frame)
                frame_counter += 1
                
                # İlerleme (Log kirliliği yapmaması için her %25'te bir bilgi verir)
                if frame_counter % (total_frames // 4 + 1) == 0:
                     print(f"   [İşçi-{worker_id}] %{(frame_counter / total_frames)*100:.0f} Tamamlandı...")

            writer.release()
            print(f"✅ [İşçi-{worker_id}] BİTTİ: {filepath}")
            
            # RAM temizliği
            del frames
            
        except Exception as e:
            print(f"❌ [İşçi-{worker_id}] HATA: {e}")
        
        SAVE_QUEUE.task_done()

# --- İŞÇİLERİ BAŞLAT ---
# Belirlenen sayı kadar Thread oluşturup sonsuz döngüye sokuyoruz
for i in range(WORKER_COUNT):
    t = threading.Thread(target=save_worker, args=(i+1,), daemon=True)
    t.start()


# ===========================
# 3. RAM BUFFER (VERİ KAYBI SIFIR)
# ===========================
class InMemoryTrack:
    def __init__(self, track, client_id):
        self.track = track
        self.client_id = client_id
        self.safe_id = re.sub(r'[^a-zA-Z0-9_\-]', '_', client_id)
        
        self.frames_buffer = [] 
        self.is_recording = False
        self.task = asyncio.create_task(self.consume_frames()) 

    async def consume_frames(self):
        while True:
            try:
                frame = await self.track.recv()
                if self.is_recording:
                    try:
                        # RAM'e BGR formatında, sıkıştırmadan atıyoruz (Maksimum Hız)
                        img = frame.to_ndarray(format="bgr24")
                        self.frames_buffer.append(img)
                    except Exception:
                        pass 
            except Exception:
                break

    def start_recording(self):
        if self.is_recording: return
        self.frames_buffer = []
        self.is_recording = True
        print(f"\n   🔴 RAM REC: {self.client_id}")

    def stop_recording(self):
        if not self.is_recording: return

        self.is_recording = False
        captured_count = len(self.frames_buffer)
        
        print(f"   ⏹️ RAM STOP: {self.client_id} | {captured_count} Kare")

        if captured_count > 0:
            # HEDEF: FULL HD (1920x1080)
            target_w = 1920
            target_h = 1080
            
            filename = f"{self.safe_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            filepath = os.path.join(VIDEO_DIR, filename)

            # Paketi hazırla ve kuyruğa fırlat
            # Hangi işçi boşsa o kapacak
            job_package = {
                "filepath": filepath,
                "frames": self.frames_buffer, 
                "width": target_w,
                "height": target_h,
                "fps": 30.0
            }
            SAVE_QUEUE.put(job_package)
        else:
            print(f"⚠️ {self.client_id} boş veri.")

# ===========================
# 4. SERVER MANAGMENT
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
                local_track = InMemoryTrack(track, raw_id)
                self.device_tracks[raw_id] = local_track

        @pc.on("connectionstatechange")
        async def on_connection_change():
            if pc.connectionState in ["failed", "closed"]:
                print(f"\n❌ Ayrıldı: {raw_id}")
                await self.cleanup(pc, raw_id)

        await pc.setRemoteDescription(offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        return web.json_response({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})

    async def cleanup(self, pc, client_id):
        self.pcs.discard(pc)
        if client_id in self.device_tracks:
            self.device_tracks[client_id].stop_recording()
            del self.device_tracks[client_id]
        await pc.close()

    async def handle_candidate(self, request):
        return web.Response(status=200)

    async def handle_main_pc_connect(self, request):
        print("\n💻 MAIN PC BAĞLANDI")
        return web.Response(text="OK")

    async def start_record(self, request):
        print("\n🚀 START: RAM Kaydı Başladı...")
        count = 0
        if not self.device_tracks: return web.json_response({"status": "failed"})
        for track in self.device_tracks.values():
            track.start_recording()
            count += 1
        return web.json_response({"status": "started", "device_count": count})

    async def stop_record(self, request):
        print("\n🛑 STOP: Videolar İşleniyor (Paralel Mod)...")
        for track in self.device_tracks.values():
            track.stop_recording()
        return web.json_response({"status": "queued"})

    async def list_videos(self, request):
        files = os.listdir(VIDEO_DIR)
        return web.json_response(files)

    async def download_video(self, request):
        name = request.match_info.get('name', request.rel_url.query.get("file"))
        if not name: return web.Response(status=400, text="Eksik")
        path = os.path.join(VIDEO_DIR, os.path.basename(name))
        print(f"💻 İndirme: {name}")
        if os.path.exists(path): return web.FileResponse(path)
        return web.Response(status=404, text="Bulunamadı")

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

    print("\n🚀 SERVER (MULTI-THREAD & ULTRA KALİTE) - HAZIR")
    print(f"⚙️ Aktif İşçi Sayısı: {WORKER_COUNT}")
    print("---------------------------------------------------")
    web.run_app(app, host="192.168.1.4", port=8080, access_log=None)

if __name__ == "__main__":
    main()