package com.example.android.wonderfulapp.image.colorCorrection

import android.graphics.Bitmap

abstract class Filter {
    abstract suspend fun apply(curBitmap: Bitmap): Bitmap
}