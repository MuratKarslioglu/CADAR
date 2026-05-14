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
# 1. AYARLAR VE LOGLAMA
# ===========================
VIDEO_DIR = "recordings"
os.makedirs(VIDEO_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Server")

# ===========================
# 2. NON-BLOCKING VIDEO KAYDEDİCİ
# ===========================
class AsyncVideoWriter:
    """
    Bu sınıf, disk yazma işlemlerini ana döngüden (Main Loop) ayırır.
    Görüntüler bir kuyruğa atılır, arka plandaki işçi (thread) bunları diske yazar.
    Böylece server asla donmaz.
    """
    def __init__(self, filepath, width, height, fps=20.0):
        self.filepath = filepath
        self.queue = queue.Queue()
        self.running = True
        self.width = width
        self.height = height
        self.fps = fps
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
        logger.info(f"💾 Kayıt Başlatıldı: {filepath}")

    def write_frame(self, frame_bgr):
        if self.running:
            self.queue.put(frame_bgr)

    def stop(self):
        self.running = False
        self.thread.join() # Thread'in bitmesini bekle
        logger.info(f"🛑 Kayıt Tamamlandı: {self.filepath}")

    def _worker(self):
        # VideoWriter'ı Thread içinde oluşturuyoruz
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        writer = cv2.VideoWriter(self.filepath, fourcc, self.fps, (self.width, self.height))

        while True:
            try:
                # Kuyruktan frame al (timeout ile döngüyü kontrol et)
                frame = self.queue.get(timeout=1)
            except queue.Empty:
                if not self.running:
                    break
                continue
            
            if frame is not None:
                writer.write(frame)
                self.queue.task_done()
        
        writer.release()

# ===========================
# 3. STREAM TRACK YÖNETİCİSİ
# ===========================
class DeviceTrack(VideoStreamTrack):
    """
    Her Android cihaz için oluşturulan sanal kanal.
    Görüntüyü alır ve eğer kayıt aktifse 'AsyncVideoWriter'a gönderir.
    """
    def __init__(self, track, client_id):
        super().__init__()
        self.track = track
        self.client_id = client_id
        self.recorder = None # Kayıtçı (başlangıçta yok)

    async def recv(self):
        frame = await self.track.recv()

        # Kayıt aktifse frame'i kopyalayıp yazıcıya gönder
        if self.recorder and self.recorder.running:
            # WebRTC Frame -> Numpy Array (RGB) -> BGR
            img = frame.to_ndarray(format="bgr24")
            self.recorder.write_frame(img)

        # Frame'i olduğu gibi geri döndür (Forwarding için gerekli)
        return frame

    def start_recording(self):
        if self.recorder and self.recorder.running:
            return # Zaten kayıt yapıyor

        filename = f"{self.client_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.avi"
        path = os.path.join(VIDEO_DIR, filename)
        
        # Varsayılan çözünürlük (Android'den gelen ilk frame'de güncellenebilir ama
        # burada standart bir boyut başlatıyoruz, dinamik yapı biraz daha komplekstir)
        self.recorder = AsyncVideoWriter(path, 640, 480) # 640x480 varsayılan

    def stop_recording(self):
        if self.recorder:
            self.recorder.stop()
            self.recorder = None

# ===========================
# 4. SERVER MİMARİSİ (STATE MANAGEMENT)
# ===========================
class MediaServer:
    def __init__(self):
        self.pcs = set() # Peer Connections
        self.device_tracks = {} # {client_id: DeviceTrack_Instance}

    async def handle_offer(self, request):
        params = await request.json()
        client_id = params.get("client_id", str(uuid.uuid4()))
        sdp = params["sdp"]
        type_ = params["type"]

        offer = RTCSessionDescription(sdp=sdp, type=type_)
        pc = RTCPeerConnection()
        self.pcs.add(pc)

        logger.info(f"🔗 Yeni Bağlantı: {client_id}")

        @pc.on("track")
        async def on_track(track):
            if track.kind == "video":
                logger.info(f"📹 Video Akışı Algılandı: {client_id}")
                # Android'den gelen track'i sarmalıyoruz
                local_track = DeviceTrack(track, client_id)
                self.device_tracks[client_id] = local_track
                
                # Eğer anlık izleme yapılacaksa pc.addTrack(local_track) denilebilir.
                # Şu an sadece serverda işliyoruz, geri göndermiyoruz (Blackhole).
                # Ancak WebRTC'nin akması için bir 'alıcı' gibi davranmalıyız.
                
                # Trick: Track'i tüketmek için boş bir döngü yerine
                # MediaRecorder kullanılabilir ama biz manuel işliyoruz.
                while True:
                    try:
                        await local_track.recv()
                    except Exception:
                        break

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            if pc.connectionState in ["failed", "closed"]:
                logger.warning(f"❌ Bağlantı Koptu: {client_id}")
                await self.cleanup_client(client_id, pc)

        await pc.setRemoteDescription(offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        return web.json_response({
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type
        })

    async def cleanup_client(self, client_id, pc):
        self.pcs.discard(pc)
        if client_id in self.device_tracks:
            self.device_tracks[client_id].stop_recording() # Kaydı güvenli kapat
            del self.device_tracks[client_id]

    # --- MAIN PC KOMUTLARI ---

    async def start_all_recordings(self, request):
        """Tüm bağlı cihazlarda kaydı başlatır."""
        count = 0
        for client_id, track in self.device_tracks.items():
            track.start_recording()
            count += 1
        return web.json_response({"status": "started", "device_count": count})

    async def stop_all_recordings(self, request):
        """Tüm kayıtları durdurur."""
        for track in self.device_tracks.values():
            track.stop_recording()
        return web.json_response({"status": "stopped"})

    async def list_files(self, request):
        files = os.listdir(VIDEO_DIR)
        return web.json_response({"files": files})
    
    async def delete_file(self, request):
        # Dosya silme endpoint'i
        data = await request.json()
        filename = data.get("filename")
        if not filename:
            return web.json_response({"error": "Dosya adı gerekli"}, status=400)
        
        path = os.path.join(VIDEO_DIR, filename)
        if os.path.exists(path):
            os.remove(path)
            return web.json_response({"status": "deleted", "file": filename})
        return web.json_response({"error": "Dosya bulunamadı"}, status=404)

# ===========================
# 5. UYGULAMA BAŞLATMA
# ===========================
async def index(request):
    return web.Response(text="WebRTC Media Server Running...")

def main():
    server = MediaServer()
    app = web.Application()

    # Route Tanımları
    app.router.add_get("/", index)
    app.router.add_post("/offer", server.handle_offer)
    
    # Main PC Kontrol Endpointleri
    app.router.add_post("/control/start_all", server.start_all_recordings)
    app.router.add_post("/control/stop_all", server.stop_all_recordings)
    
    # Dosya İşlemleri
    app.router.add_get("/files", server.list_files)
    app.router.add_post("/files/delete", server.delete_file)
    app.router.add_static("/download", VIDEO_DIR) # Videoları indirmek için

    print("🚀 Server Başlatılıyor... http://0.0.0.0:8080")
    web.run_app(app, host="0.0.0.0", port=8080)

if __name__ == "__main__":
    main()