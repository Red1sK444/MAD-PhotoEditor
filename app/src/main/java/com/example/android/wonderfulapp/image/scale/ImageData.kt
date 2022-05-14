package com.example.android.wonderfulapp.image.scale

class ImageData(val pixels: IntArray, val width: Int, val height: Int) {
    fun setPixel(pixel: Int, x: Int, y: Int) {
        pixels[y * width + x] = pixel
    }

    fun getPixel(x: Int, y: Int): Int {
        return pixels[y * width + x]
    }
}
