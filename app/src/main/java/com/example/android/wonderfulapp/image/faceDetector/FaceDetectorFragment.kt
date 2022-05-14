package com.example.android.wonderfulapp.image.faceDetector

import android.content.Context
import android.graphics.Bitmap
import android.os.Bundle
import android.view.View
import android.widget.Toast
import by.kirich1409.viewbindingdelegate.viewBinding
import com.example.android.wonderfulapp.R
import com.example.android.wonderfulapp.databinding.FragmentFaceDetectorBinding
import com.example.android.wonderfulapp.image.AlgorithmFragment
import com.example.android.wonderfulapp.image.ImageManager
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.opencv.android.BaseLoaderCallback
import org.opencv.android.LoaderCallbackInterface
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.core.MatOfRect
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.imgproc.Imgproc
import org.opencv.objdetect.CascadeClassifier
import java.io.File
import java.io.FileOutputStream

class
FaceDetectorFragment : AlgorithmFragment(R.layout.fragment_face_detector) {

    private val viewBinding by viewBinding(FragmentFaceDetectorBinding::bind, R.id.rootLayout)

    companion object {
        val TAG: String = FaceDetectorFragment::class.java.simpleName
        fun newInstance() = FaceDetectorFragment()
    }

    private lateinit var classifier: CascadeClassifier
    private lateinit var faceDetections: MatOfRect
    private lateinit var cascadeFile: File

    private lateinit var matImagePreview: Mat
    private lateinit var matImage: Mat

    private var allContentLoaded = false

    private val mLoaderCallback by lazy {
        (object : BaseLoaderCallback(requireContext()) {
            override fun onManagerConnected(status: Int) {
                when (status) {
                    LoaderCallbackInterface.SUCCESS -> {
                        val iStream =
                            resources.openRawResource(R.raw.haarcascade_frontalface_alt2)

                        val cascadeDir: File =
                            requireActivity().getDir("cascade", Context.MODE_PRIVATE)
                        cascadeFile = File(cascadeDir, "haarcascade_frontalface_alt2.xml")

                        val foStream = FileOutputStream(cascadeFile)

                        val buffer = ByteArray(4096)
                        var bytesRead: Int = iStream.read(buffer)

                        while (bytesRead != -1) {
                            foStream.write(buffer, 0, bytesRead)
                            bytesRead = iStream.read(buffer)
                        }

                        iStream.close()
                        foStream.close()

                        classifier = CascadeClassifier(cascadeFile.absolutePath)
                        if (!classifier.empty()) {
                            cascadeDir.delete()
                        }

                        matImagePreview = Mat()
                        matImage = Mat()
                        updateUtils()
                    }
                    else -> {
                        super.onManagerConnected(status)
                    }
                }
            }
        })
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        imageView = viewBinding.imageView
        loadingSpinner = viewBinding.loadingSpinner

        super.onViewCreated(view, savedInstanceState)
        super.setAcceptDeclineButtonsListeners(viewBinding.acceptBtn, viewBinding.declineBtn)
    }

    override fun acceptButtonListener() {
        enableLoadingUi()

        uiScope.launch {
            detectFaces(matImage)?.let { ImageManager.image = it }
            super.acceptButtonListener()
            disableLoadingUi()
        }
    }

    override fun changeAllInteractionState(state: Boolean) {
        viewBinding.acceptBtn.isEnabled = state
        viewBinding.declineBtn.isEnabled = state
    }

    override fun initFragment() {
        super.initFragment()
        enableLoadingUi()

        uiScope.launch {
            if (allContentLoaded) {
                detectFaces(matImagePreview)?.let { updateThumbnail(it) }
            }
            disableLoadingUi()
        }
    }

    private suspend fun detectFaces(image: Mat): Bitmap? {
        return withContext(Dispatchers.IO) {
            if (!classifier.empty()) {
                faceDetections = MatOfRect()
                classifier.detectMultiScale(image, faceDetections)

                drawRectangles(image)
                imageWithFrames(image)
            } else {
                null
            }
        }
    }

    private fun drawRectangles(image: Mat) {
        for (rect in faceDetections.toArray()) {
            Imgproc.rectangle(
                image,
                Point(rect.x.toDouble(), rect.y.toDouble()),
                Point(
                    (rect.x + rect.width).toDouble(),
                    (rect.y + rect.height).toDouble()
                ),
                Scalar(255.0, 255.0, 255.0),
                3
            )
        }
    }

    private fun imageWithFrames(image: Mat): Bitmap {
        val imageWithRectangles =
            Bitmap.createBitmap(image.cols(), image.rows(), Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(image, imageWithRectangles)
        return imageWithRectangles
    }

    override fun onResume() {
        super.onResume()
        if (!OpenCVLoader.initDebug()) {
            OpenCVLoader.initAsync(
                OpenCVLoader.OPENCV_VERSION_3_0_0,
                requireContext(),
                mLoaderCallback
            )
        } else {
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS)
        }

        uiScope.launch {
            if (!allContentLoaded) {
                detectFaces(matImagePreview)?.let { updateThumbnail(it) }
                allContentLoaded = true
            }
        }
    }

    override fun updateThumbnail(image: Bitmap) {
        super.updateThumbnail(image)
        Toast.makeText(
            requireContext(),
            getString(R.string.faces_detected, faceDetections.toArray().size),
            Toast.LENGTH_LONG
        ).show()
        updateUtils()
    }

    private fun updateUtils() {
        Utils.bitmapToMat(ImageManager.thumbnail, matImagePreview)
        Utils.bitmapToMat(ImageManager.image, matImage)
    }
}