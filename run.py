# run.py
from app import create_app

app = create_app(light=False)

if __name__ == "__main__":
    # 先关闭 debug 和 reloader，保证单进程，方便排错
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)
