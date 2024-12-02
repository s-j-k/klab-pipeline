import os
import sys
from preprocessing import ScanboxViewer
from PyQt5.QtWidgets import QApplication


file_path = "/Volumes/Andrew's External Hard Drive/Data/sk83/session0/raw.sk83.003.001/sk83_003_001.sbx"
file_extension = os.path.split(file_path)[-1].split('.')[-1]

app = QApplication(sys.argv)
if file_extension == 'sbx':
    ScanboxViewer(file_path, app=app)
    sys.exit(app.exec_())
else:
    raise NotImplementedError()
