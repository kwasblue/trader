from PySide6 import QtGui, QtWidgets

# =====================================================
# THEME (high-contrast dark)
# =====================================================


def apply_dark_palette(app: QtWidgets.QApplication) -> None:
    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.Window, QtGui.QColor("#0a0a0a"))
    palette.setColor(QtGui.QPalette.WindowText, QtGui.QColor("#e5e5e5"))
    palette.setColor(QtGui.QPalette.Base, QtGui.QColor("#0f0f0f"))
    palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor("#111111"))
    palette.setColor(QtGui.QPalette.ToolTipBase, QtGui.QColor("#0f0f0f"))
    palette.setColor(QtGui.QPalette.ToolTipText, QtGui.QColor("#e5e5e5"))
    palette.setColor(QtGui.QPalette.Text, QtGui.QColor("#e5e5e5"))
    palette.setColor(QtGui.QPalette.Button, QtGui.QColor("#141414"))
    palette.setColor(QtGui.QPalette.ButtonText, QtGui.QColor("#f0f0f0"))
    palette.setColor(QtGui.QPalette.BrightText, QtGui.QColor("#ff4d4f"))
    palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor("#2563eb"))
    palette.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor("#ffffff"))
    app.setPalette(palette)
