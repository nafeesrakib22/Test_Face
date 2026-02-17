import atexit
from backend.services.camera_services import vs, load_resources

def startup():
    print("🚀 Initializing Hardware and AI Models...")
    load_resources()
    vs.start()

@atexit.register
def shutdown():
    print("🛑 Releasing Hardware...")
    vs.stop()