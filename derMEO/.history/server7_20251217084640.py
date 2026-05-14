import os
import json
import uuid
import logging
from datetime import datetime
from aiohttp import web, WSMsgType

# ===========================
# 1. AYARLAR
# ===========================
VIDEO_DIR = "videos"
os.makedirs(VIDEO_DIR, exist_ok=True)

# Log seviyesini ayarlayalım
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MediaServer")

# ===========================
# 2. SERVER YÖNETİCİSİ
# ===========================
class CommandServer:
    def __init__(self):
        # Bağlı olan Android cihazların WebSocket listesi
        self.connected_devices = {} 

    async def handle_websocket(self, request):
        """
        Android cihazlar buraya bağlanır ve emir bekler.
        """
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        # Cihaz bağlanınca ona geçici bir ID verelim (veya handshake ile alabilirsin)
        client_id = str(uuid.uuid4())[:8]
        self.connected_devices[client_id] = ws
        logger.info(f"📱 Cihaz Bağlandı: {client_id} (Toplam: {len(self.connected_devices)})")

        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    # Android'den "Ben hazırım" vb. mesajlar gelirse buraya düşer
                    if data.get("type") == "pong":
                        print(f"📡 {client_id} hala hatta.")
                elif msg.type == WSMsgType.ERROR:
                    logger.error(f"❌ {client_id} Bağlantı hatası: {ws.exception()}")
        finally:
            # Bağlantı koptuğunda listeden sil
            if client_id in self.connected_devices:
                del self.connected_devices[client_id]
            logger.info(f"❌ Cihaz Ayrıldı: {client_id}")

        return ws

    async def trigger_start(self, request):
        """
        Main PC bu endpoint'e istek atınca tüm Androidlere 'RECORD_START' emri gider.
        """
        if not self.connected_devices:
            return web.json_response({"status": "error", "message": "Hiçbir cihaz bağlı değil!"})

        logger.info("🚀 TÜM CİHAZLARA 'KAYIT BAŞLA' EMRİ GÖNDERİLİYOR...")
        
        command = json.dumps({"command": "START_RECORDING"})
        count = 0
        
        # Bağlı herkese gönder
        for client_id, ws in self.connected_devices.items():
            try:
                await ws.send_str(command)
                count += 1
            except Exception as e:
                logger.error(f"⚠️ {client_id} cihazına gönderilemedi: {e}")

        return web.json_response({"status": "started", "device_count": count})

    async def trigger_stop(self, request):
        """
        Main PC istek atınca tüm Androidlere 'RECORD_STOP' emri gider.
        Androidler kaydı bitirip dosyayı upload etmeye başlar.
        """
        logger.info("🛑 TÜM CİHAZLARA 'KAYIT DUR VE YÜKLE' EMRİ GÖNDERİLİYOR...")
        
        command = json.dumps({"command": "STOP_AND_UPLOAD"})
        count = 0

        for client_id, ws in self.connected_devices.items():
            try:
                await ws.send_str(command)
                count += 1
            except Exception as e:
                logger.error(f"⚠️ {client_id} cihazına gönderilemedi: {e}")

        return web.json_response({"status": "stopped", "device_count": count})

    async def handle_file_upload(self, request):
        """
        Android cihazlar videoyu kaydettikten sonra buraya POST ederler.
        Multipart/form-data olarak gelir.
        """
        logger.info("📥 Bir dosya yüklemesi başladı...")
        
        reader = await request.multipart()
        
        # Dosya alanını bul
        field = await reader.next()
        if field.name != 'video':
            return web.Response(status=400, text="Form key 'video' olmalı.")

        # Dosya ismini oluştur (Cihazın gönderdiği isim veya server tarafından)
        filename = field.filename
        if not filename:
            filename = f"upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        
        # Güvenli dosya yolu
        filename = os.path.basename(filename)
        filepath = os.path.join(VIDEO_DIR, filename)

        # Dosyayı diske yaz (Chunk chunk okuyarak RAM şişmesini engeller)
        size = 0
        with open(filepath, 'wb') as f:
            while True:
                chunk = await field.read_chunk()  # Varsayılan 8KB chunk
                if not chunk:
                    break
                f.write(chunk)
                size += len(chunk)

        logger.info(f"✅ Dosya Kaydedildi: {filename} ({size / (1024*1024):.2f} MB)")
        return web.json_response({"status": "success", "file": filename})

    async def list_videos(self, request):
        files = os.listdir(VIDEO_DIR)
        return web.json_response(files)

    async def download_video(self, request):
        name = request.match_info.get('name')
        if not name: return web.Response(status=400, text="Eksik dosya adı")
        
        path = os.path.join(VIDEO_DIR, name)
        if os.path.exists(path):
            return web.FileResponse(path)
        return web.Response(status=404, text="Dosya bulunamadı")

# ===========================
# 3. UYGULAMA BAŞLATMA
# ===========================
def main():
    app = web.Application()
    server = CommandServer()

    # WebSocket (Cihazların sürekli bağlı kalacağı hat)
    app.router.add_get("/ws", server.handle_websocket)

    # Komutlar (Main PC'nin tetikleyeceği endpointler)
    app.router.add_post("/start_record", server.trigger_start)
    app.router.add_post("/stop_record", server.trigger_stop)

    # Dosya Yükleme (Android'in video bitince atacağı yer)
    app.router.add_post("/upload", server.handle_file_upload)

    # Main PC için listeleme ve indirme
    app.router.add_get("/list_videos", server.list_videos)
    app.router.add_get("/download_video/{name}", server.download_video)

    print("\n🚀 KOMUT & DEPO SERVER HAZIR")
    print("---------------------------------------------------")
    print("📡 Androidler için WS:  ws://IP:8080/ws")
    print("📤 Android Upload URL:  http://IP:8080/upload")
    print("💻 Main PC Başlatma:    POST http://IP:8080/start_record")
    print("💻 Main PC Durdurma:    POST http://IP:8080/stop_record")
    
    # max_request_size: Video uploadları için limiti artırıyoruz (örneğin 500MB)
    web.run_app(app, host="0.0.0.0", port=8080, client_max_size=1024**3) # 1GB Limit

if __name__ == "__main__":
    main()