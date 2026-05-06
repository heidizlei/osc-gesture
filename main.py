import asyncio
import threading
import json
from classes.app import OSCGestureApp

if __name__ == "__main__":
    app = OSCGestureApp(ip="100.101.30.29")
    app.run()
