package com.example.android.wonderfulapp.image.colorCorrection

enum class Filters(val filter: Filter?) {
    BLACK_WHITE(BlackWhiteFilter()),
    CHANGE_CONTRAST(ChangeContrastFilter()),
    NEGATIVE(NegativeFilter()),
    PATRIOTIC(PatrioticFilter()),
    SEPIA(SepiaFilter()),
    SKETCH(SketchFilter()),
    STACK_BLUR(StackBlurFilter()),
    NONE(null)
}