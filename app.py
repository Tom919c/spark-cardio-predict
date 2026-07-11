import os
import threading
import webbrowser

from src import create_app


app = create_app()


def open_browser():
    url = "http://127.0.0.1:5000/dashboard"
    try:
        os.startfile(url)
    except AttributeError:
        webbrowser.open(url)


if __name__ == "__main__":
    threading.Timer(1.5, open_browser).start()
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
