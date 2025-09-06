import logging
import socket
import os

from routes import app

logger = logging.getLogger(__name__)


@app.route('/', methods=['GET'])
def default_route():
    return 'Python Template'


logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter(
    '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

if __name__ == "__main__":
    logging.info("Starting application ...")
    
    # Get host and port from environment variables or use defaults
    host = os.getenv('HOST', '0.0.0.0')  # 0.0.0.0 allows external connections
    port = int(os.getenv('PORT', 8080))
    
    # Check if port is available
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind((host, port))
        port = sock.getsockname()[1]
        sock.close()
        logging.info(f"Server will start on {host}:{port}")
        app.run(host=host, port=port, debug=False)
    except OSError as e:
        sock.close()
        logging.error(f"Cannot bind to {host}:{port} - {e}")
        # Try with a random port
        app.run(host=host, port=0, debug=False)
