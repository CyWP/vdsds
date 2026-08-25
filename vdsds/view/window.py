import numpy as np

from typing import Tuple

from PySide6.QtCore import QObject, Qt, Signal, QLabel, QPoint
from PySide6.QtGui import QKeyEvent, QImage, QPixmap, QMouseEvent, QWheelEvent
from PySide6.QtWidgets import QMainWindow, QSizePolicy


class AppView(QObject):
    """
    Viewer component of MVC.
    Public API exposes signals for controller connections and methods for updating the UI.
    """

    def __init__(self) -> None:
        super().__init__()
        self.window: MainWindow = MainWindow()
        self.window.setWindowTitle("VDSDS")
        self.window.show()

    def update(self, img: np.ndarray):
        """
        Update the viewport with a new image.
        Args:
            img (np.ndarray): Image to display in the viewport.
        """
        self.window.viewport.set_image(img)


class MainWindow(QMainWindow):
    key_pressed = Signal(int)
    key_released = Signal(int)

    def __init__(self):
        super().__init__()
        self.setWindowFlags(
            Qt.Window
            | Qt.CustomizeWindowHint
            | Qt.WindowMinMaxButtonsHint
            | Qt.WindowCloseButtonHint
        )
        # ----- Main window settings -----
        self.setWindowTitle("VDSDS")
        self.resize(1400, 900)  # Set reasonable default size
        self.move(200, 100)  # Place window at some position on screen

        # ----- Central viewport -----
        self.viewport = Viewport(self)
        self.viewport.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setCentralWidget(self.viewport)
        viewport_rect = self.viewport.rect()  # QRect(0,0,width,height)
        global_pos = self.viewport.mapTo(self, viewport_rect.topLeft())

        # Optional: start maximized now that geometry is set
        self.showMaximized()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        viewport_rect = self.viewport.rect()
        global_pos = self.viewport.mapTo(self, viewport_rect.topLeft())

    def keyPressEvent(self, event: QKeyEvent) -> None:
        self.key_pressed.emit(event.key())
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent) -> None:
        self.key_released.emit(event.key())
        super().keyReleaseEvent(event)


class Viewport(QLabel):
    """
    A QLabel-based viewport that displays images from NumPy arrays.
    Automatically scales images to fit the available space and emits mouse
    interaction signals.
    """

    # Left mouse signals
    left_clicked: Signal = Signal(int, int)
    left_pressed: Signal = Signal(int, int)
    left_released: Signal = Signal(int, int)
    dragged: Signal = Signal(int, int, int, int)  # (start_x, start_y, curr_x, curr_y)
    drag_direction: Signal = Signal(int, int)  # delta_x, delta_y

    # Right mouse signals
    right_clicked: Signal = Signal(int, int)
    right_pressed: Signal = Signal(int, int)
    right_released: Signal = Signal(int, int)

    # Scroll signal
    scrolled: Signal = Signal(int)  # positive = up, negative = down

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("viewport")
        self.setFocusPolicy(Qt.StrongFocus)  # needed for key events
        self._pixmap: QPixmap | None = None
        self._drag_start: QPoint | None = None

    def showEvent(self, event):
        """Initialize the viewport with a solid background color once the widget has a valid size."""
        super().showEvent(event)

    def set_image(self, array: np.ndarray):
        """Update viewport with an image from a NumPy array."""
        # breakpoint()
        if array.ndim == 2:  # grayscale
            h, w = array.shape
            bytes_per_line = w
            qimage = QImage(array.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
        elif array.ndim == 3 and array.shape[2] == 3:  # RGB
            h, w, _ = array.shape
            bytes_per_line = 3 * w
            qimage = QImage(array.data, w, h, bytes_per_line, QImage.Format_RGB888)
        elif array.ndim == 3 and array.shape[2] == 4:  # RGBA
            h, w, _ = array.shape
            bytes_per_line = 4 * w
            qimage = QImage(array.data, w, h, bytes_per_line, QImage.Format_RGBA8888)
        else:
            raise ValueError("Unsupported image format")

        self._pixmap = QPixmap.fromImage(qimage)
        self.setPixmap(
            self._pixmap.scaled(
                self.width(), self.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )

    def get_dimensions(self) -> Tuple[int]:
        return self.height(), self.width()

    def resizeEvent(self, event):
        """Automatically scale the pixmap when the widget is resized."""
        super().resizeEvent(event)
        if self._pixmap:
            self.setPixmap(
                self._pixmap.scaled(
                    self.width(),
                    self.height(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
            )

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            self._drag_start = self._drag_last = event.pos()
            self.left_pressed.emit(event.pos().x(), event.pos().y())
        elif event.button() == Qt.RightButton:
            self.right_pressed.emit(event.pos().x(), event.pos().y())
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._drag_start is not None:  # only for left button
            start = self._drag_start
            last = self._drag_last
            current = event.pos()
            self.dragged.emit(start.x(), start.y(), current.x(), current.y())
            self.drag_direction.emit(current.x() - last.x(), current.y() - last.y())
            self._drag_last = current
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton and self._drag_start is not None:
            self.left_released.emit(event.pos().x(), event.pos().y())
            if (event.pos() - self._drag_start).manhattanLength() < 5:
                self.left_clicked.emit(event.pos().x(), event.pos().y())
            self._drag_start = self._drag_last = None
        elif event.button() == Qt.RightButton:
            self.right_released.emit(event.pos().x(), event.pos().y())
            self.right_clicked.emit(event.pos().x(), event.pos().y())
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event: QWheelEvent) -> None:
        delta = event.angleDelta().y()
        self.scrolled.emit(delta)
        super().wheelEvent(event)
