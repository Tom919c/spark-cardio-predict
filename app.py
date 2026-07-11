import os
import threading
import webbrowser

from src import create_app


app = create_app()


def open_browser():
    """启动后自动打开浏览器到仪表盘页面。"""
    url = f"http://{app.config['HOST']}:{app.config['PORT']}/dashboard"
    try:
        os.startfile(url)
    except AttributeError:
        webbrowser.open(url)


if __name__ == "__main__":
    threading.Timer(1.5, open_browser).start()
    app.run(
        host=app.config["HOST"],
        port=app.config["PORT"],
        debug=app.config.get("DEBUG", False),
        use_reloader=False,
    )
