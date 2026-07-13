import os
from typing import Optional
from dataclasses import dataclass

from dotenv import load_dotenv

# Optional imports guarded at runtime to keep platform flexibility
try:
	import pywhatkit
except Exception:  # pragma: no cover - optional
	pywhatkit = None

try:
	from twilio.rest import Client
except Exception:  # pragma: no cover - optional
	Client = None


load_dotenv()


@dataclass
class AlertConfig:
	# pywhatkit
	whatsapp_phone: Optional[str] = os.getenv("WHATSAPP_PHONE")  # like +91XXXXXXXXXX
	# Twilio
	twilio_sid: Optional[str] = os.getenv("TWILIO_ACCOUNT_SID")
	twilio_token: Optional[str] = os.getenv("TWILIO_AUTH_TOKEN")
	twilio_from: Optional[str] = os.getenv("TWILIO_WHATSAPP_FROM")  # like whatsapp:+14155238886
	twilio_to: Optional[str] = os.getenv("TWILIO_WHATSAPP_TO")  # like whatsapp:+91XXXXXXXXXX


class AlertSender:
	def __init__(self, config: Optional[AlertConfig] = None):
		self.config = config or AlertConfig()

	def send_whatsapp_pywhatkit(self, message: str) -> bool:
		if pywhatkit is None or not self.config.whatsapp_phone:
			return False
		try:
			# pywhatkit sends at a scheduled minute; to send immediately we schedule one minute from now.
			# WARNING: Requires WhatsApp Web open and may need GUI automation permissions.
			import datetime
			now = datetime.datetime.now()
			hour, minute = now.hour, (now.minute + 1) % 60
			pywhatkit.sendwhatmsg(self.config.whatsapp_phone, message, hour, minute, wait_time=10, tab_close=True, close_time=3)
			return True
		except Exception:
			return False

	def send_whatsapp_twilio(self, message: str, media_url: Optional[str] = None) -> bool:
		if Client is None or not (self.config.twilio_sid and self.config.twilio_token and self.config.twilio_from and self.config.twilio_to):
			return False
		try:
			client = Client(self.config.twilio_sid, self.config.twilio_token)
			kwargs = {
				"from_": self.config.twilio_from,
				"to": self.config.twilio_to,
				"body": message,
			}
			if media_url:
				kwargs["media_url"] = [media_url]
			client.messages.create(**kwargs)
			return True
		except Exception:
			return False

	def send_alert(self, message: str, media_url: Optional[str] = None) -> str:
		# Try Twilio first (more reliable, headless), then fall back to pywhatkit
		if self.send_whatsapp_twilio(message, media_url=media_url):
			return "twilio"
		if self.send_whatsapp_pywhatkit(message):
			return "pywhatkit"
		return "none"
