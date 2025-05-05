from PyQt5.QtCore import QPropertyAnimation, QEasingCurve, QSequentialAnimationGroup, QParallelAnimationGroup, QPoint, QSize

def fade_in(widget, duration=300, ease=QEasingCurve.OutCubic):
    """Fade in animation with customizable easing curve."""
    animation = QPropertyAnimation(widget, b"windowOpacity")
    animation.setDuration(duration)
    animation.setStartValue(0)
    animation.setEndValue(1)
    animation.setEasingCurve(ease)
    return animation

def fade_out(widget, duration=300, ease=QEasingCurve.InCubic):
    """Fade out animation with customizable easing curve."""
    animation = QPropertyAnimation(widget, b"windowOpacity")
    animation.setDuration(duration)
    animation.setStartValue(1)
    animation.setEndValue(0)
    animation.setEasingCurve(ease)
    return animation

def slide_in(widget, direction='right', distance=100, duration=400, ease=QEasingCurve.OutBack):
    """
    Slide in animation with customizable direction, distance, and easing curve.
    Directions: 'right', 'left', 'up', 'down'
    """
    start_pos = widget.pos()
    
    if direction == 'right':
        widget.move(start_pos.x() - distance, start_pos.y())
    elif direction == 'left':
        widget.move(start_pos.x() + distance, start_pos.y())
    elif direction == 'down':
        widget.move(start_pos.x(), start_pos.y() - distance)
    elif direction == 'up':
        widget.move(start_pos.x(), start_pos.y() + distance)
        
    animation = QPropertyAnimation(widget, b"pos")
    animation.setDuration(duration)
    animation.setStartValue(widget.pos())
    animation.setEndValue(start_pos)
    animation.setEasingCurve(ease)
    return animation

def slide_out(widget, direction='right', distance=100, duration=400, ease=QEasingCurve.InBack):
    """
    Slide out animation with customizable direction, distance, and easing curve.
    Directions: 'right', 'left', 'up', 'down'
    """
    start_pos = widget.pos()
    end_pos = QPoint(start_pos)
    
    if direction == 'right':
        end_pos.setX(start_pos.x() + distance)
    elif direction == 'left':
        end_pos.setX(start_pos.x() - distance)
    elif direction == 'down':
        end_pos.setY(start_pos.y() + distance)
    elif direction == 'up':
        end_pos.setY(start_pos.y() - distance)
        
    animation = QPropertyAnimation(widget, b"pos")
    animation.setDuration(duration)
    animation.setStartValue(start_pos)
    animation.setEndValue(end_pos)
    animation.setEasingCurve(ease)
    return animation

def scale(widget, start_scale=1.0, end_scale=1.1, duration=200, ease=QEasingCurve.OutQuad):
    """Scale animation for hovering effects."""
    geo = widget.geometry()
    center = geo.center()
    
    animation = QPropertyAnimation(widget, b"geometry")
    animation.setDuration(duration)
    animation.setStartValue(geo)
    
    new_width = int(geo.width() * end_scale)
    new_height = int(geo.height() * end_scale)
    new_geo = geo
    new_geo.setWidth(new_width)
    new_geo.setHeight(new_height)
    new_geo.moveCenter(center)
    
    animation.setEndValue(new_geo)
    animation.setEasingCurve(ease)
    return animation

def pulse(widget, scale_factor=1.05, duration=500):
    """Creates a pulsing animation that scales up and down."""
    geo = widget.geometry()
    center = geo.center()
    
    scale_up = QPropertyAnimation(widget, b"geometry")
    scale_up.setDuration(duration // 2)
    scale_up.setStartValue(geo)
    
    new_width = int(geo.width() * scale_factor)
    new_height = int(geo.height() * scale_factor)
    new_geo = geo
    new_geo.setWidth(new_width)
    new_geo.setHeight(new_height)
    new_geo.moveCenter(center)
    
    scale_up.setEndValue(new_geo)
    scale_up.setEasingCurve(QEasingCurve.OutQuad)
    
    scale_down = QPropertyAnimation(widget, b"geometry")
    scale_down.setDuration(duration // 2)
    scale_down.setStartValue(new_geo)
    scale_down.setEndValue(geo)
    scale_down.setEasingCurve(QEasingCurve.InQuad)
    
    animation_group = QSequentialAnimationGroup()
    animation_group.addAnimation(scale_up)
    animation_group.addAnimation(scale_down)
    
    return animation_group

def combo_animation(animations):
    """Combine multiple animations to run in parallel."""
    group = QParallelAnimationGroup()
    for animation in animations:
        group.addAnimation(animation)
    return group