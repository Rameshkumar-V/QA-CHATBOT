"""Entry point for the Windows executable. Starts a local Streamlit server."""
import os
import sys
from pathlib import Path
from streamlit.web import cli as stcli

if __name__ == '__main__':
    app_file = Path(getattr(sys, '_MEIPASS', Path(__file__).resolve().parent)) / 'app.py'
    os.environ.setdefault('STREAMLIT_BROWSER_GATHER_USAGE_STATS', 'false')
    sys.argv = ['streamlit', 'run', str(app_file), '--server.address=127.0.0.1',
                '--server.headless=false', '--browser.gatherUsageStats=false']
    sys.exit(stcli.main())
