# run.py
import os
import threading
import webbrowser
from app import create_app

# 保持你原来的参数
app = create_app(light=False)

def _print_banner(host: str, port: int) -> str:
    url = f"http://{host}:{port}"
    banner = (
        "\n────────────────────────────────────────\n"
        f"  Image Drive running at: {url}\n"
        f"  Studio:    {url}/\n"
        f"  Analytics: {url}/analytics\n"
        "────────────────────────────────────────\n"
    )
    # 同时用 logger 和 stdout 打印，确保你在任何环境都能看到
    try:
        app.logger.info(banner)
    finally:
        print(banner, flush=True)
    return url

if __name__ == "__main__":
    # 支持用环境变量覆盖地址/端口和是否自动打开浏览器
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "5000"))
    open_browser = os.environ.get("OPEN_BROWSER", "0").lower() in ("1", "true", "yes")

    url = _print_banner(host, port)

    if open_browser:
        # 延时 1 秒打开浏览器，避免服务尚未就绪
        threading.Timer(1.0, lambda: webbrowser.open_new(url)).start()

    # 关闭 debug 和 reloader，保证单进程，便于排错（你原本的设定）
    app.run(host=host, port=port, debug=False, use_reloader=False)
