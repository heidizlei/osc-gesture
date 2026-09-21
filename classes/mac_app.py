"""Native macOS shell for the bundled .app.

Launched from Finder the app has no terminal, and without an AppKit event loop
nothing answers the Dock's Quit or Cmd-Q, so the only way out would be Force
Quit -- with the camera on until then. This runs NSApplication on the main
thread (AppKit requires it) and the gesture loop on a worker thread, which
gives the app:

- Quit from the Dock, the menu bar, Cmd-Q or Ctrl+C, each of which stops the
  gesture loop and releases the camera before exiting;
- the controls re-opened in the browser when the app is re-opened (Dock click,
  or double-clicking it again), since closing the tab doesn't stop the app;
- an alert instead of silently vanishing if the loop dies, e.g. because
  another copy already holds the web UI port.

main.py only uses this for the web UI in a frozen build. From source the loop
keeps the main thread and Ctrl+C stops it as before.
"""

import errno
import signal
import sys
import threading
import traceback
import webbrowser

from AppKit import NSAlert, NSApplication, NSMenu, NSMenuItem, NSTerminateNow
from Foundation import NSObject
from PyObjCTools import AppHelper, MachSignals

APP_NAME = "OSC Gesture"

# How long quitting waits for the gesture loop to stop and release the camera.
_STOP_TIMEOUT_S = 3.0


def _menu_item(title, action, key, target=None):
    item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(title, action, key)
    if target is not None:
        item.setTarget_(target)
    return item


class OSCGestureAppDelegate(NSObject):
    # run_as_mac_app sets .app, .worker and .url before the event loop starts.

    def applicationShouldHandleReopen_hasVisibleWindows_(self, sender, has_windows):
        self.openControls_(None)
        return False

    def applicationDockMenu_(self, sender):
        menu = NSMenu.alloc().init()
        menu.addItem_(_menu_item("Open Controls in Browser", "openControls:", "", self))
        return menu

    def applicationShouldTerminate_(self, sender):
        self.app.running = False
        self.worker.join(_STOP_TIMEOUT_S)
        # Terminating exits without Python's shutdown, so flush output that
        # is still buffered (stdout redirected to a file, say).
        sys.stdout.flush()
        sys.stderr.flush()
        return NSTerminateNow

    def openControls_(self, sender):
        webbrowser.open(self.url)


def _main_menu(delegate):
    app_menu = NSMenu.alloc().init()
    app_menu.addItem_(_menu_item("Open Controls in Browser", "openControls:", "o", delegate))
    app_menu.addItem_(NSMenuItem.separatorItem())
    app_menu.addItem_(_menu_item(f"Quit {APP_NAME}", "terminate:", "q"))
    app_item = NSMenuItem.alloc().init()
    app_item.setSubmenu_(app_menu)
    menubar = NSMenu.alloc().init()
    menubar.addItem_(app_item)
    return menubar


def _alert_loop_failed(error, http_port):
    if isinstance(error, OSError) and error.errno == errno.EADDRINUSE:
        detail = (f"Port {http_port} is already in use. "
                  f"Is {APP_NAME} already running?")
    else:
        detail = str(error) or type(error).__name__
    alert = NSAlert.alloc().init()
    alert.setMessageText_(f"{APP_NAME} stopped")
    alert.setInformativeText_(detail)
    NSApplication.sharedApplication().activateIgnoringOtherApps_(True)
    alert.runModal()


def run_as_mac_app(app, http_host, http_port, open_browser):
    """Run the web-UI gesture loop under a native event loop. Never returns."""
    ns_app = NSApplication.sharedApplication()
    browse_host = "127.0.0.1" if http_host in ("0.0.0.0", "") else http_host

    def loop_finished(error):
        if error is not None:
            _alert_loop_failed(error, http_port)
        ns_app.terminate_(None)

    def gesture_loop():
        error = None
        try:
            app.run(ui="web", http_host=http_host, http_port=http_port,
                    open_browser=open_browser)
        except Exception as e:
            traceback.print_exc()
            error = e
        # The loop also ends on the web UI's quit control; either way the app
        # goes with it.
        AppHelper.callAfter(loop_finished, error)

    delegate = OSCGestureAppDelegate.alloc().init()
    delegate.app = app
    delegate.url = f"http://{browse_host}:{http_port}"
    delegate.worker = threading.Thread(target=gesture_loop, name="gesture-loop",
                                       daemon=True)
    ns_app.setDelegate_(delegate)
    ns_app.setMainMenu_(_main_menu(delegate))

    # Ctrl+C, when the bundle is started from a terminal. Python's own signal
    # handlers can't run while AppKit holds the main thread; MachSignals
    # delivers the signal through the run loop instead.
    for signum in (signal.SIGINT, signal.SIGTERM):
        MachSignals.signal(signum, lambda _signum: ns_app.terminate_(None))

    delegate.worker.start()
    AppHelper.runEventLoop()
