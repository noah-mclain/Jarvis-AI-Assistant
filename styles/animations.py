from PyQt5.QtCore import QPropertyAnimation, QEasingCurve

def fade_in(widget, duration=300):
    animation = QPropertyAnimation(widget, b"windowOpacity")
    animation.setDuration(duration)
    animation.setStartValue(0)
    animation.setEndValue(1)
    animation.setEasingCurve(QEasingCurve.OutQuad)
    return animation

def slide_in(widget, direction='right', duration=400):
    start_pos = widget.pos()
    if direction == 'right':
        widget.move(start_pos.x() + 100, start_pos.y())
    else:
        widget.move(start_pos.x() - 100, start_pos.y())
        
    animation = QPropertyAnimation(widget, b"pos")
    animation.setDuration(duration)
    animation.setStartValue(widget.pos())
    animation.setEndValue(start_pos)
    animation.setEasingCurve(QEasingCurve.OutBack)
    return animation