"""Reusable Qt widgets: image view, mask painter, PIL conversion."""
from __future__ import annotations

from PIL import Image
from PySide6.QtCore import QPoint, QPointF, QRect, Qt, Signal
from PySide6.QtGui import QColor, QImage, QMouseEvent, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QLabel, QSizePolicy, QWidget


def pil_to_qimage(img: Image.Image) -> QImage:
    img = img.convert("RGBA")
    data = img.tobytes("raw", "RGBA")
    return QImage(data, img.width, img.height, img.width * 4, QImage.Format_RGBA8888).copy()


def qimage_to_pil(qimg: QImage) -> Image.Image:
    qimg = qimg.convertToFormat(QImage.Format_RGBA8888)
    ptr = qimg.constBits()
    return Image.frombuffer("RGBA", (qimg.width(), qimg.height()), bytes(ptr), "raw", "RGBA",
                            qimg.bytesPerLine(), 1).copy()


class ImageView(QLabel):
    """Shows a PIL image scaled to fit, keeping proportions."""

    def __init__(self, placeholder: str = ""):
        super().__init__(placeholder)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(200, 200)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.setStyleSheet("QLabel { background: #1e1e1e; color: #888; }")
        self._pixmap: QPixmap | None = None

    def set_image(self, img: Image.Image | None) -> None:
        self._pixmap = QPixmap.fromImage(pil_to_qimage(img)) if img is not None else None
        self._rescale()

    def resizeEvent(self, e):  # noqa: N802
        super().resizeEvent(e)
        self._rescale()

    def _rescale(self) -> None:
        if self._pixmap is None:
            self.setPixmap(QPixmap())
            return
        self.setPixmap(self._pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))


class MaskCanvas(QWidget):
    """Source image with a paintable mask (left button paints, right button erases)."""

    changed = Signal()

    def __init__(self):
        super().__init__()
        self.setMinimumSize(200, 200)
        self.setMouseTracking(True)
        self.brush = 48
        self.mask_enabled = True
        self._image: QImage | None = None
        self._mask: QImage | None = None
        self._last: QPointF | None = None
        self._erase = False
        self._cursor: QPointF | None = None

    # -- data
    def set_image(self, img: Image.Image | None) -> None:
        if img is None:
            self._image = self._mask = None
        else:
            self._image = pil_to_qimage(img.convert("RGB"))
            self._mask = QImage(self._image.size(), QImage.Format_Grayscale8)
            self._mask.fill(0)
        self.update()
        self.changed.emit()

    def image(self) -> Image.Image | None:
        return qimage_to_pil(self._image).convert("RGB") if self._image is not None else None

    def mask(self) -> Image.Image | None:
        if self._mask is None:
            return None
        return qimage_to_pil(self._mask).convert("L")

    def clear_mask(self) -> None:
        if self._mask is not None:
            self._mask.fill(0)
            self.update()

    def invert_mask(self) -> None:
        if self._mask is not None:
            self._mask.invertPixels()
            self.update()

    # -- geometry
    def _target(self) -> QRect:
        if self._image is None:
            return QRect()
        s = self._image.size().scaled(self.size(), Qt.KeepAspectRatio)
        return QRect((self.width() - s.width()) // 2, (self.height() - s.height()) // 2, s.width(), s.height())

    def _to_image(self, p: QPointF) -> QPointF:
        t = self._target()
        sx = self._image.width() / max(1, t.width())
        return QPointF((p.x() - t.x()) * sx, (p.y() - t.y()) * sx)

    # -- painting
    def paintEvent(self, _):  # noqa: N802
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#1e1e1e"))
        if self._image is None:
            p.setPen(QColor("#888"))
            p.drawText(self.rect(), Qt.AlignCenter, "Загрузите исходное изображение\nили отправьте сюда результат")
            return
        t = self._target()
        p.drawImage(t, self._image)
        if self.mask_enabled and self._mask is not None:
            overlay = QImage(self._mask.size(), QImage.Format_ARGB32)
            overlay.fill(QColor(255, 40, 80))
            overlay.setAlphaChannel(self._mask)
            p.setOpacity(0.5)
            p.drawImage(t, overlay)
            p.setOpacity(1.0)
            if self._cursor is not None:
                r = self.brush / 2 * t.width() / self._image.width()
                p.setPen(QPen(QColor("white"), 1))
                p.drawEllipse(self._cursor, r, r)

    def _stroke(self, a: QPointF, b: QPointF) -> None:
        p = QPainter(self._mask)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(QPen(QColor(0, 0, 0) if self._erase else QColor(255, 255, 255), self.brush,
                      Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        p.drawLine(self._to_image(a), self._to_image(b))
        p.end()
        self.update()

    def mousePressEvent(self, e: QMouseEvent):  # noqa: N802
        if not (self.mask_enabled and self._mask is not None):
            return
        self._erase = e.button() == Qt.RightButton
        self._last = e.position()
        self._stroke(self._last, self._last + QPoint(0, 0))

    def mouseMoveEvent(self, e: QMouseEvent):  # noqa: N802
        self._cursor = e.position()
        if self._last is not None:
            self._stroke(self._last, e.position())
            self._last = e.position()
        self.update()

    def mouseReleaseEvent(self, _):  # noqa: N802
        self._last = None
        self.changed.emit()

    def leaveEvent(self, _):  # noqa: N802
        self._cursor = None
        self.update()

    def wheelEvent(self, e):  # noqa: N802
        step = 8 if e.angleDelta().y() > 0 else -8
        self.brush = max(4, min(400, self.brush + step))
        self.update()
