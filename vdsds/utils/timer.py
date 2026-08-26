import threading
import time
import traceback


class TimedJob:
    def __init__(self):
        self.thread = None
        self.running = False
        self.interval = None

    def start(self, action: callable = None, frequency: int = None):
        if self.running:
            self.stop()
        self.action = self.action if action is None else action
        if frequency:
            self.interval = 1.0 / frequency
        elif self.interval is None:
            self.interval = 1.0 / 60
        self.running = True
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    def loop(self):
        while self.running:
            start_time = time.time()
            try:
                self.action()
            except Exception:
                traceback.print_exc()
                self.running = False
                break
            elapsed = time.time() - start_time
            sleep_time = self.interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join()
