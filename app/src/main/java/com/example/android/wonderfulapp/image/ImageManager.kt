package com.example.android.wonderfulapp.image

import android.content.ContentResolver
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri

object ImageManager {
    const val imageMaxSize = 4000 * 2000
    const val imageMinSize = 128 * 128
    private const val imageMaxWidth = 4000

    private const val thumbnailMaxSize = 720 * 576
    private const val thumbnailMaxWidth = 720

    var imagePath: Uri? = null
    var image: Bitmap? = null
    var thumbnail: Bitmap? = null

    fun updateImage(newPath: Uri, contentResolver: ContentResolver) {
        newPath.let {
            imagePath = newPath
            image = getScaledBitmap(getDecodedImage(newPath, contentResolver), imageMaxSize, imageMaxWidth)
            thumbnail = getScaledBitmap(image, thumbnailMaxSize, thumbnailMaxWidth)
        }
    }

    fun reDownscaleThumbnail() {
        thumbnail = getScaledBitmap(thumbnail, thumbnailMaxSize, thumbnailMaxWidth)
    }

    private fun getDecodedImage(path: Uri, contentResolver: ContentResolver): Bitmap? {
        val imgStream = contentResolver.openInputStream(path)
        return BitmapFactory.decodeStream(imgStream)
    }

    private fun getScaledBitmap(bitmap: Bitmap?, maxSize: Int, maxWidth: Int): Bitmap? {
        bitmap?.let {
            if (it.width * it.height > maxSize) {
                return it.scale(maxWidth)
            }
            return it
        }
        return null
    }

    private fun Bitmap?.scale(maxWidth: Int): Bitmap? {
        this?.let {
            val ratio: Float = it.width.toFloat() / it.height.toFloat()
            val height = (maxWidth / ratio).toInt()

            return Bitmap.createScaledBitmap(it, maxWidth, height, true)
        }

        return null
    }
}