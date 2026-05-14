import os
import json
import logging
import uuid
from datetime import datetime
from aiohttp import web, WSMsgType

# ===========================
# 1. AYARLAR
# ===========================
VIDEO_DIR = "videos"
os.makedirs(VIDEO_DIR, exist_ok=True)

# Sadece önemli bilgileri görelim
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("MediaServer")

class MediaServer:
    def __init__(self):
        # Bağlı cihazları tutar { "Cihaz_1": ws_obj, "Main_PC": ws_obj }
        self.connections = {}
        self.device_counter = 0

    # ===========================
    # 2. WEBSOCKET YÖNETİMİ (Emir-Komuta)
    # ===========================
    async def handle_ws(self, request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        # Cihazın kim olduğunu sorgudan veya otomatik alalım
        # Örn: ws://IP:8080/ws?type=main veya ws://IP:8080/ws?type=phone
        client_type = request.query.get('type', 'phone')
        
        if client_type == 'main':
            client_name = "💻 MAIN PC"
        else:
            self.device_counter += 1
            client_name = f"📱 Cihaz_{self.device_counter}"

        self.connections[client_name] = ws
        logger.info(f"✅ {client_name} BAĞLANDI (Toplam: {len(self.connections)})")

        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    logger.info(f"📩 {client_name} Mesajı: {data}")
        finally:
            if client_name in self.connections:
                del self.connections[client_name]
            logger.info(f"❌ {client_name} AYRILDI")
        
        return ws

    # ===========================
    # 3. KAYIT EMİRLERİ (PC'den Gelir)
    # ===========================
    async def start_record(self, request):
        """Main PC burayı tetikleyince tüm telefonlara 'Kaydet' der"""
        logger.info("\n🚀 [KOMUT] KAYIT BAŞLATILIYOR...")
        command = json.dumps({"command": "START_RECORDING"})
        
        count = 0
        for name, ws in self.connections.items():
            if "Cihaz" in name:
                await ws.send_str(command)
                count += 1
        
        return web.json_response({"status": "command_sent", "target_count": count})

    async def stop_record(self, request):
        """Main PC burayı tetikleyince telefonlar kaydı durdurur ve upload eder"""
        logger.info("\n🛑 [KOMUT] KAYIT DURDUR VE YÜKLE!")
        command = json.dumps({"command": "STOP_AND_UPLOAD"})
        
        count = 0
        for name, ws in self.connections.items():
            if "Cihaz" in name:
                await ws.send_str(command)
                count += 1
                
        return web.json_response({"status": "command_sent", "target_count": count})

    # ===========================
    # 4. DOSYA YÜKLEME (Android'den Gelir)
    # ===========================
    async def handle_upload(self, request):
        """Android cihazlar kaydı bitirince dosyayı bu endpoint'e POST eder"""
        reader = await request.multipart()
        field = await reader.next()
        
        if field.name != 'video':
            return web.Response(status=400, text="Hatalı dosya anahtarı")

        filename = field.filename or f"upload_{uuid.uuid4().hex[:6]}.mp4"
        filepath = os.path.join(VIDEO_DIR, filename)

        size = 0
        with open(filepath, 'wb') as f:
            while True:
                chunk = await field.read_chunk()
                if not chunk: break
                f.write(chunk)
                size += len(chunk)

        logger.info(f"📥 DOSYA ALINDI: {filename} ({size / (1024*1024):.2f} MB)")
        return web.json_response({"status": "success", "received": filename})

    # ===========================
    # 5. PC İÇİN LİSTELEME VE İNDİRME
    # ===========================
    async def list_videos(self, request):
        files = os.listdir(VIDEO_DIR)
        return web.json_response(files)

    async def download_video(self, request):
        name = request.match_info.get('name')
        path = os.path.join(VIDEO_DIR, name)
        if os.path.exists(path):
            return web.FileResponse(path)
        return web.Response(status=404, text="Dosya yok")

# ===========================
# 6. SERVER START
# ===========================
def main():
    server = MediaServer()
    app = web.Application(client_max_size=1024**3) # 1GB Limit

    # WebSocket
    app.router.add_get("/ws", server.handle_ws)

    # Kontrol Uçları (PC için)
    app.router.add_post("/start_record", server.start_record)
    app.router.add_post("/stop_record", server.stop_record)

    # Veri Uçları (Telefon için)
    app.router.add_post("/upload", server.handle_upload)

    # Dosya Yönetimi (PC için)
    app.router.add_get("/list_videos", server.list_videos)
    app.router.add_get("/download/{name}", server.download_video)

    print("\n" + "="*50)
    print("🚀 MEO-SERVER: KAYIT VE DEPO SİSTEMİ HAZIR")
    print("="*50)
    print("📡 Androidler: ws://192.168.1.4:8080/ws?type=phone")
    print("💻 Main PC:    ws://192.168.1.4:8080/ws?type=main")
    print("---------------------------------------------------\n")

    web.run_app(app, host="0.0.0.0", port=8080, access_log=None)

if __name__ == "__main__":
    main()