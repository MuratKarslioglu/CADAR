import asyncio
import os
import uuid
import threading
import queue
import re
import cv2
import logging
from datetime import datetime
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription

# ===========================
# 1. AYARLAR
# ===========================
VIDEO_DIR = "videos"
os.makedirs(VIDEO_DIR, exist_ok=True)

logging.getLogger("aiohttp").setLevel(logging.WARNING)
logging.getLogger("aiortc").setLevel(logging.WARNING)

# ===========================
# 2. GLOBAL KAYIT KUYRUĞU (SIRALI KAYIT MERKEZİ)
# ===========================
# Bütün cihazlar çekimi bitirince verilerini buraya atar.
# Tek bir işçi (thread) buradaki işleri sırayla alıp diske yazar.
SAVE_QUEUE = queue.Queue()


def global_save_worker():
    """
    Arka planda sürekli çalışır. Kuyruğa bir video paketi düştüğünde
    onu alır ve diske yazar. İşlem bitmeden diğerine geçmez.
    """
    print("⚙️ Disk Yazma İşçisi Hazır (Sıralı Kayıt Modu)")

    while True:
        # Kuyruktan iş bekle (Bloklayıcı)
        job = SAVE_QUEUE.get()

        filepath = job['filepath']
        frames = job['frames']
        width = job['width']
        height = job['height']
        fps = job['fps']

        print(f"\n💾 DİSK YAZMA BAŞLADI: {filepath}")
        print(f"   📦 Toplam Kare: {len(frames)} | Kuyrukta Bekleyen: {SAVE_QUEUE.qsize()}")

        try:
            # Video Writer oluştur
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(filepath, fourcc, fps, (width, height))

            frame_counter = 0
            total_frames = len(frames)

            for frame in frames:
                # Gerekirse resize (Güvenlik)
                if frame.shape[1] != width or frame.shape[0] != height:
                    frame = cv2.resize(frame, (width, height))

                writer.write(frame)
                frame_counter += 1

                # İlerleme çubuğu gibi her %10'da bir nokta bas
                if frame_counter % (total_frames // 10 + 1) == 0:
                    print("#", end="", flush=True)

            writer.release()
            print(f"\n✅ DİSK YAZMA BİTTİ: {filepath}")

            # RAM'i temizle (Python Garbage Collector'a yardım et)
            del frames

        except Exception as e:
            print(f"❌ KAYIT HATASI: {e}")

        # Kuyruğa işin bittiğini bildir
        SAVE_QUEUE.task_done()


# İşçiyi başlat (Daemon: Ana program kapanınca bu da kapanır)
threading.Thread(target=global_save_worker, daemon=True).start()


# ===========================
# 3. RAM BUFFER TRACK (HAFİZA KAYDEDİCİ)
# ===========================
class InMemoryTrack:
    def __init__(self, track, client_id):
        self.track = track
        self.client_id = client_id
        self.safe_id = re.sub(r'[^a-zA-Z0-9_\-]', '_', client_id)

        # --- RAM BUFFER ---
        self.frames_buffer = []  # Görüntüler burada birikecek (RAM)
        self.is_recording = False
        self.task = asyncio.create_task(self.consume_frames())

    async def consume_frames(self):
        while True:
            try:
                frame = await self.track.recv()

                # Eğer kayıt başladıysa RAM listesine ekle
                if self.is_recording:
                    try:
                        # WebRTC frame -> Numpy Array (BGR)
                        img = frame.to_ndarray(format="bgr24")
                        self.frames_buffer.append(img)
                    except Exception:
                        pass
            except Exception:
                break

    def start_recording(self):
        if self.is_recording:
            return

        # Listeyi sıfırla
        self.frames_buffer = []
        self.is_recording = True
        print(f"\n   🔴 RAM KAYDI BAŞLADI: {self.client_id} (Veriler RAM'e yazılıyor...)")

    def stop_recording(self):
        if not self.is_recording:
            return

        self.is_recording = False
        captured_count = len(self.frames_buffer)

        print(f"   ⏹️ RAM KAYDI DURDU: {self.client_id} | Toplanan Kare: {captured_count}")

        if captured_count > 0:
            # Kayıt için gerekli bilgileri hazırla
            # Son karenin boyutlarını al (Referans olarak)
            last_frame = self.frames_buffer[-1]
            h, w = last_frame.shape[:2]

            # Çift sayı kuralı (MP4 için)
            safe_w = w if w % 2 == 0 else w - 1
            safe_h = h if h % 2 == 0 else h - 1

            filename = f"{self.safe_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            filepath = os.path.join(VIDEO_DIR, filename)

            # GLOBAL KUYRUĞA GÖNDER
            job_package = {
                "filepath": filepath,
                "frames": self.frames_buffer,  # DİKKAT: RAM'deki listeyi gönderiyoruz
                "width": safe_w,
                "height": safe_h,
                "fps": 30.0
            }

            SAVE_QUEUE.put(job_package)
            print(f"   ⏳ KUYRUĞA EKLENDİ: {self.client_id} -> Sırasını bekliyor...")

            # Listeyi burada sıfırlamıyoruz! Worker işini bitirince silecek.
            # self.frames_buffer = [] # BURADA SİLME!
        else:
            print(f"⚠️ HATA: {self.client_id} için hiç kare yakalanamadı.")

# ===========================
# 4. SERVER YÖNETİMİ
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

        return web.json_response({
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type
        })

    async def cleanup(self, pc, client_id):
        self.pcs.discard(pc)
        # Bağlantı kopsa bile kaydı durdurup kuyruğa atar
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
        print("\n🚀 START: RAM'e Kayıt Başlıyor...")
        count = 0
        if not self.device_tracks:
            return web.json_response({"status": "failed"})

        for track in self.device_tracks.values():
            track.start_recording()
            count += 1
        return web.json_response({"status": "started", "device_count": count})

    async def stop_record(self, request):
        print("\n🛑 STOP: Kayıt Bitti -> Disk Yazma Kuyruğuna Aktarılıyor...")
        for track in self.device_tracks.values():
            track.stop_recording()

        # Kullanıcıya hemen cevap dönüyoruz, arka planda yazma devam ediyor
        return web.json_response({"status": "queued", "message": "Videolar sırayla diske yazılıyor..."})

    async def list_videos(self, request):
        files = os.listdir(VIDEO_DIR)
        return web.json_response(files)

    async def download_video(self, request):
        name = request.match_info.get('name', request.rel_url.query.get("file"))
        if not name:
            return web.Response(status=400, text="Eksik isim")

        name = os.path.basename(name)
        path = os.path.join(VIDEO_DIR, name)

        print(f"💻 İndirme İsteği: {name}")
        if os.path.exists(path):
            return web.FileResponse(path)
        return web.Response(status=404, text="Bulunamadı (Henüz yazılıyor olabilir)")


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

    print("\n🚀 SERVER HAZIR (RAM BUFFER & QUEUE MODE) - http://0.0.0.0:8080")
    print("⚠️ UYARI: Bu modda videolar önce RAM'e kaydedilir.")
    print("⚠️ Uzun süreli kayıtlarda RAM dolabilir!")
    print("---------------------------------------------------")
    web.run_app(app, host="0.0.0.0", port=8080, access_log=None)


if __name__ == "__main__":
    main()
