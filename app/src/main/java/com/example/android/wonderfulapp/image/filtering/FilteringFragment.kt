package com.example.android.wonderfulapp.image.filtering

import android.graphics.Bitmap
import android.graphics.PixelFormat
import android.os.Bundle
import android.view.View
import android.widget.Toast
import by.kirich1409.viewbindingdelegate.viewBinding
import com.example.android.wonderfulapp.R
import com.example.android.wonderfulapp.databinding.FragmentFilteringBinding
import com.example.android.wonderfulapp.image.AlgorithmFragment
import com.example.android.wonderfulapp.image.ImageManager
import com.example.android.wonderfulapp.image.scale.ScaleAlgorithm
import kotlinx.coroutines.launch

class FilteringFragment : AlgorithmFragment(R.layout.fragment_filtering) {

    private val viewBinding by viewBinding(FragmentFilteringBinding::bind, R.id.rootLayout)
    private var affineMatrices: MutableList<FloatArray> = mutableListOf()

    companion object {
        val TAG: String = FilteringFragment::class.java.simpleName
        fun newInstance() = FilteringFragment()
    }

    private fun initCanvas() {
        ImageManager.thumbnail?.let {
            viewBinding.filteringSurfaceView.visibility = View.VISIBLE
            viewBinding.filteringSurfaceView.holder.setFixedSize(it.width, it.height)
        }
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        super.setAcceptDeclineButtonsListeners(viewBinding.acceptBtn, viewBinding.declineBtn)

        initCanvas()

        viewBinding.pointsStateBtn.setOnClickListener { changePointsState() }

        viewBinding.clearBtn.setOnClickListener { clearCurrentPoints() }

        viewBinding.filterBtn.setOnClickListener {
            transformImage()
        }
    }

    private fun changePointsState() {
        val newState = Points.toggleState()
        viewBinding.pointsStateText.text = getStateText(newState)
        viewBinding.filteringSurfaceView.updateCanvas()
    }

    private fun clearCurrentPoints() {
        Points.clearCurrentStatePoints()
        viewBinding.filteringSurfaceView.updateCanvas()
    }

    private fun getStateText(state: PointsState): String {
        return when (state) {
            PointsState.START -> {
                getString(R.string.start_points)
            }
            PointsState.FINISH -> {
                getString(R.string.finish_points)
            }
        }
    }

    override fun acceptButtonListener() {
        enableLoadingUi()

        ImageManager.thumbnail?.let {
            updateThumbnail(it)
        }

        uiScope.launch {
            ImageManager.image =
                ImageManager.image?.let { image ->
                    val filteringAlgorithm = FilteringAlgorithm(image)
                    val transformedImg =
                        filteringAlgorithm.transformImage(affineMatrices[0], affineMatrices[1])

                    getFilteredImage(transformedImg)
                }

            super.acceptButtonListener()
            disableLoadingUi()
        }
    }

    override fun declineButtonListener() {
        transformedThumbnail?.let {
            updateThumbnail(it)
        }
        super.declineButtonListener()
    }

    private fun transformImage() {
        if (!allPointsSet()) {
            Toast.makeText(
                requireContext(),
                getString(R.string.not_enough_points),
                Toast.LENGTH_SHORT
            ).show()
        } else {
            enableLoadingUi()

            uiScope.launch {
                ImageManager.thumbnail?.let { image ->
                    val filteringAlgorithm = FilteringAlgorithm(image)
                    updateAffineMatrices(filteringAlgorithm)

                    val transformedImg = filteringAlgorithm.getTransformedImage()
                    ImageManager.thumbnail = getFilteredImage(transformedImg)
                    Points.clearAllPoints()

                    viewBinding.filteringSurfaceView.updateCanvas()
                }

                disableLoadingUi()
            }
        }
    }

    private suspend fun getFilteredImage(img: Bitmap): Bitmap {
        return ScaleAlgorithm(img).scaleImage(1F)
    }

    private fun allPointsSet(): Boolean {
        return Points.getPoints(PointsState.START).size == 3 && Points.getPoints(PointsState.FINISH).size == 3
    }

    private fun updateAffineMatrices(filteringAlgorithm: FilteringAlgorithm) {
        val newAffineMatrices = filteringAlgorithm.getAffineMatrices()
        if (affineMatrices.isEmpty()) {
            affineMatrices.add(newAffineMatrices[0])
            affineMatrices.add(newAffineMatrices[1])
        } else {
            affineMatrices[0] =
                multiply2dMatrices(affineMatrices[0], newAffineMatrices[0])
            affineMatrices[1] = add2dMatrices(affineMatrices[1], newAffineMatrices[1])
        }
    }

    private fun multiply2dMatrices(matrix1: FloatArray, matrix2: FloatArray): FloatArray {
        val newMatrix = FloatArray(4)
        newMatrix[0] = matrix1[0] * matrix2[0] + matrix1[1] * matrix2[2]
        newMatrix[1] = matrix1[0] * matrix2[1] + matrix1[1] * matrix2[3]
        newMatrix[2] = matrix1[2] * matrix2[0] + matrix1[3] * matrix2[2]
        newMatrix[3] = matrix1[1] * matrix2[2] + matrix1[3] * matrix2[3]

        return newMatrix
    }

    private fun add2dMatrices(matrix1: FloatArray, matrix2: FloatArray): FloatArray {
        val newMatrix = FloatArray(2)
        newMatrix[0] = matrix1[0] + matrix2[0]
        newMatrix[1] = matrix1[1] + matrix2[1]

        return newMatrix
    }

    override fun changeAllInteractionState(state: Boolean) {
        viewBinding.declineBtn.isEnabled = state
        viewBinding.acceptBtn.isEnabled = state
        viewBinding.pointsStateBtn.isEnabled = state
        viewBinding.clearBtn.isEnabled = state
        viewBinding.filterBtn.isEnabled = state
    }

    override fun onHiddenChanged(hidden: Boolean) {
        viewBinding.filteringSurfaceView.holder.setFormat(if (!hidden) PixelFormat.OPAQUE else PixelFormat.TRANSPARENT)
        if (!hidden) {
            initCanvas()
        }
    }
}