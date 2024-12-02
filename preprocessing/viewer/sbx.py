from sbxreader.reg import *

import pyqtgraph as pg

from PyQt5.QtWidgets import (QWidget, QFormLayout, QCheckBox, QSlider, QLabel, QMainWindow, QDockWidget)
from PyQt5.QtCore import Qt, QTimer
from sbxreader import sbx_memmap


pg.setConfigOptions(imageAxisOrder='row-major')


class ImageViewerWidget(pg.GraphicsLayoutWidget):
    def __init__(self, parent, sbxdata):
        super(ImageViewerWidget, self).__init__()
        self.sbxdata = sbxdata
        self.parent = parent
        frame = np.array(self.sbxdata[0, :, :, :, :])
        self.nplanes = self.sbxdata.shape[1]
        self.nchannels = self.sbxdata.shape[2]
        self.string = '# {0}'
        self.stringShift = '# {0} - shift ({1:1.1f},{2:1.1f})'
        p1 = self.addPlot(title="")
        p1.getViewBox().invertY(True)
        p1.hideAxis('left')
        p1.hideAxis('bottom')
        nplanes, nchannels, H, W = frame.shape
        positions = [[int(np.mod(i, 2)) * W,
                      int(i / 2) * H] for i in range(self.nplanes)]
        self.imgview = [pg.ImageItem() for i in range(self.nplanes)]
        for p, img in zip(positions, self.imgview):
            img.setPos(*p)

        for img in self.imgview:
            p1.addItem(img)

        self.plane = 0
        self.register = False
        self.references = [[None for ichannel in range(self.nchannels)] for iplane in range(self.nplanes)]
        self.update(0)
        self.show()

    def update(self, nframe):
        stack = np.array(self.sbxdata[nframe, :, :, :, self.sbxdata.ndeadcols:]).astype(np.float32)
        if self.register:
            for iplane in range(self.nplanes):
                for ichannel in range(self.nchannels):
                    if self.references[iplane][ichannel] is None:
                        self.references[iplane][ichannel] = np.squeeze(
                            np.array(
                                self.sbxdata[:256,
                                iplane,
                                ichannel,
                                :,
                                self.sbxdata.ndeadcols:]).mean(axis=0)).squeeze()
                    shift = registration_upsample(self.references[iplane][ichannel][100:-100, 100:-100],
                                                  stack[iplane][ichannel][100:-100, 100:-100])
                    stack[iplane][ichannel] = shift_image(stack[iplane][ichannel], shift)
                if iplane == 0:
                    # set the shift value for text
                    pass

        for iplane in range(self.nplanes):
            img = np.squeeze(stack[iplane])
            levels = self.parent.levels_frac * np.array([0, 2 ** 16])
            if self.nchannels > 1:
                img = np.zeros([stack.shape[2], stack.shape[3], 3], dtype=np.uint8)
                img[:, :, 0] = 255 * stack[iplane][1] / (2 ** 16 - 1)  # R
                img[:, :, 1] = 255 * stack[iplane][0] / (2 ** 16 - 1)  # G
                if self.nchannels > 2:
                    img[:, :, 2] = 255 * stack[iplane][2] / (2 ** 16 - 1)  # B
                levels = self.parent.levels_frac * np.array([0, 255])

            self.imgview[iplane].setImage(img, autoLevels=False, levels=levels)
        self.lastnFrame = nframe


class ControlWidget(QWidget):
    def __init__(self, parent):
        super(ControlWidget, self).__init__()
        self.parent = parent
        form = QFormLayout()
        self.setLayout(form)
        self.frameSlider = QSlider(Qt.Horizontal)
        self.frameSlider.setValue(0)
        self.frameSlider.setMinimum(0)
        self.frameSlider.setMaximum(self.parent.mmap.shape[0] - 1)
        self.frameSlider.setSingleStep(1)
        self.frameSliderLabel = QLabel('Frame [{0}]:'.format(self.frameSlider.value()))
        self.frameSlider.valueChanged.connect(self.setFrame)
        form.addRow(self.frameSliderLabel, self.frameSlider)

        self.playback = QCheckBox()
        self.playback.setChecked(False)
        self.playback.stateChanged.connect(self.togglePlayback)
        form.addRow(QLabel("Playback: "), self.playback)
        self.register = QCheckBox()
        self.register.setChecked(False)
        self.register.stateChanged.connect(self.toggleRegister)
        form.addRow(QLabel("Register: "), self.register)

        self.levelSlider = QSlider(Qt.Horizontal)
        self.levelSlider.setValue(100)
        self.levelSlider.setMinimum(0)
        self.levelSlider.setMaximum(100)
        self.levelSlider.setSingleStep(1)
        self.levelSlider.valueChanged.connect(self.setLevelMax)
        form.addRow(QLabel('Level max'), self.levelSlider)

    def setFrame(self, value):
        self.frameSliderLabel.setText('Frame [{0}]:'.format(int(value)))
        self.parent.widgets[0].update(int(value))

    def setLevelMax(self, value):
        self.parent.levels_frac[1] = value / 100.
        self.parent.widgets[0].update(int(self.frameSlider.value()))

    def setPlane(self, value):
        iPlane = self.planeSelector.currentIndex()
        self.parent.widgets[0].plane = iPlane
        self.parent.widgets[0].update(int(self.frameSlider.value()))

    def togglePlayback(self, value):
        if value:
            self.parent.timer.start()
        else:
            self.parent.timer.stop()

    def toggleRegister(self, value):
        self.parent.widgets[0].register = value


class ScanboxViewer(QMainWindow):
    app = None

    def __init__(self, fname=None, app=None):
        super(ScanboxViewer, self).__init__()
        self.app = app
        self.filename = fname
        self.mmap = sbx_memmap(self.filename)

        nframes, nplanes, nchans, W, H = self.mmap.shape
        self.nplanes = nplanes
        self.nframes = nframes
        self.levels = np.array([0, np.percentile(np.array(self.mmap[1]), 98)])
        self.levels_frac = np.array([0., 1.])

        self.tabs = []
        self.widgets = []
        self.controlWidget = None
        self.timer = None
        self.initUI()

    def initUI(self):
        # Menu
        self.setWindowTitle("Scanbox viewer")
        self.tabs.append(QDockWidget("Frames", self))
        self.widgets.append(ImageViewerWidget(self, self.mmap))

        self.tabs[-1].setWidget(self.widgets[-1])
        self.tabs[-1].setFloating(False)
        self.addDockWidget(
            Qt.RightDockWidgetArea and Qt.TopDockWidgetArea,
            self.tabs[-1])
        c = 1
        self.controlWidget = ControlWidget(self)
        self.tabs.append(QDockWidget("Frame control", self))
        self.tabs[-1].setWidget(self.controlWidget)
        self.tabs[-1].setFloating(False)
        self.addDockWidget(Qt.TopDockWidgetArea, self.tabs[-1])
        self.timer = QTimer()
        self.timer.timeout.connect(self.timerUpdate)
        # self.timer.start(10)
        self.move(0, 0)
        self.show()

    def timerUpdate(self):
        self.controlWidget.frameSlider.setValue(np.mod(
            self.controlWidget.frameSlider.value() + 1, self.nframes))
