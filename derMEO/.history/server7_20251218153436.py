import os
import json
import uuid
import logging
from datetime import datetime
from aiohttp import web, WSMsgType

# ===========================
# 1. AYARLAR & LOGLAMA
# ===========================
VIDEO_DIR = "videos"
os.makedirs(VIDEO_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MediaServer")

class CommandServer:
    def __init__(self):
        # Cihazları ve isimlerini tutan sözlük
        self.connected_devices = {} 
        self.device_counter = 0

    async def handle_websocket(self, request):
        """
        Android cihazlar veya Main PC buraya bağlanır.
        """
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        # Cihaza isim verelim
        self.device_counter += 1
        current_device_name = f"Cihaz_{self.device_counter}"
        
        # Eğer bağlantı sorgusunda özel bir isim varsa onu kullanalım
        # Örn: ws://IP:8080/ws?name=MainPC
        custom_name = request.query.get('name')
        if custom_name:
            current_device_name = custom_name

        self.connected_devices[current_device_name] = ws
        logger.info(f"🟢 {current_device_name} BAĞLANDI (Toplam: {len(self.connected_devices)})")

        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    # Cihazdan gelen özel mesajları burada loglayabilirsin
                    logger.info(f"📩 {current_device_name} mesaj gönderdi: {data}")
        finally:
            if current_device_name in self.connected_devices:
                del self.connected_devices[current_device_name]
            logger.info(f"🔴 {current_device_name} AYRILDI")
        
        return ws

    async def handle_webrtc_signal(self, request):
        """Flutter'dan gelen /offer, /candidate, /answer istekleri"""
        path = request.path
        # İstek atan IP'ye göre log basalım ki kim olduğunu anlayalım
        client_ip = request.remote
        logger.info(f"📡 WebRTC Sinyali -> Yol: {path} | Gönderen IP: {client_ip}")
        return web.json_response({"status": "received"})

    async def trigger_start(self, request):
        """Main PC bu endpointi tetiklediğinde"""
        logger.info("💻 MAIN PC: 'Kaydı Başlat' emri verdi!")
        command = json.dumps({"command": "START_RECORDING"})
        
        count = 0
        for name, ws in self.connected_devices.items():
            if "Cihaz" in name: # Sadece cihazlara gönder, PC'ye geri gönderme
                await ws.send_str(command)
                count += 1
        
        return web.json_response({"status": "started", "sent_to": count})

    async def trigger_stop(self, request):
        """Main PC bu endpointi tetiklediğinde"""
        logger.info("💻 MAIN PC: 'Kaydı Durdur ve Yükle' emri verdi!")
        command = json.dumps({"command": "STOP_AND_UPLOAD"})
        
        count = 0
        for name, ws in self.connected_devices.items():
            if "Cihaz" in name:
                await ws.send_str(command)
                count += 1
                
        return web.json_response({"status": "stopped", "sent_to": count})

    async def handle_file_upload(self, request):
        logger.info("📥 Video yüklemesi başladı...")
        reader = await request.multipart()
        field = await reader.next()
        
        filename = field.filename or f"video_{datetime.now().strftime('%H%M%S')}.mp4"
        filepath = os.path.join(VIDEO_DIR, os.path.basename(filename))

        with open(filepath, 'wb') as f:
            while True:
                chunk = await field.read_chunk()
                if not chunk: break
                f.write(chunk)
        
        logger.info(f"✅ DOSYA ALINDI: {filename}")
        return web.json_response({"status": "success"})

# ===========================
# 3. BAŞLATMA
# ===========================
def main():
    app = web.Application()
    server = CommandServer()

    app.router.add_get("/ws", server.handle_websocket)
    app.router.add_post("/start_record", server.trigger_start)
    app.router.add_post("/stop_record", server.trigger_stop)
    app.router.add_post("/upload", server.handle_file_upload)
    
    # 404 Hatalarını önlemek için WebRTC yolları
    app.router.add_post("/offer", server.handle_webrtc_signal)
    app.router.add_post("/candidate", server.handle_webrtc_signal)
    app.router.add_post("/answer", server.handle_webrtc_signal)

    print("\n🚀 SERVER AKTİF - CİHAZLAR BEKLENİYOR")
    print("---------------------------------------------------")
    web.run_app(app, host="0.0.0.0", port=8080, client_max_size=1024**3)

if __name__ == "__main__":
    main()