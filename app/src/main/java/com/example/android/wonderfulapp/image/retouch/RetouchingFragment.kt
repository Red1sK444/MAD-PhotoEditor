package com.example.android.wonderfulapp.image.retouch

import android.os.Bundle
import android.view.MotionEvent
import android.view.View
import android.widget.SeekBar
import by.kirich1409.viewbindingdelegate.viewBinding
import com.example.android.wonderfulapp.R
import com.example.android.wonderfulapp.databinding.FragmentRetouchingBinding
import com.example.android.wonderfulapp.image.AlgorithmFragment
import com.example.android.wonderfulapp.image.ImageManager
import com.example.android.wonderfulapp.image.helpFunc.ConvertCoordinates
import com.example.android.wonderfulapp.image.helpFunc.Coordinates
import com.example.android.wonderfulapp.image.helpFunc.TextViewChanger.Companion.changeParameter
import com.example.android.wonderfulapp.image.rotating.ImageParameters
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking

class RetouchingFragment : AlgorithmFragment(R.layout.fragment_retouching) {
    val minRadius = 20.0
    val minAmount = 1.0
    private val retouchingAlgorithm = RetouchingAlgorithm()

    private val viewBinding by viewBinding(FragmentRetouchingBinding::bind, R.id.rootLayout)

    companion object {
        val TAG: String = RetouchingFragment::class.java.simpleName
        fun newInstance() = RetouchingFragment()
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        imageView = viewBinding.imageView
        loadingSpinner = viewBinding.loadingSpinner

        super.onViewCreated(view, savedInstanceState)
        super.setAcceptDeclineButtonsListeners(viewBinding.acceptBtn, viewBinding.declineBtn)

        viewBinding.radius.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onStopTrackingTouch(seekBar: SeekBar) {}
            override fun onStartTrackingTouch(seekBar: SeekBar) {}
            override fun onProgressChanged(seekBar: SeekBar, progress: Int, fromUser: Boolean) {
                changeParameter(
                    viewBinding.radiusValue,
                    minRadius.toInt() + viewBinding.radius.progress
                )
            }
        })

        viewBinding.amount.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onStopTrackingTouch(seekBar: SeekBar) {}
            override fun onStartTrackingTouch(seekBar: SeekBar) {}
            override fun onProgressChanged(seekBar: SeekBar, progress: Int, fromUser: Boolean) {
                changeParameter(
                    viewBinding.amountValue,
                    minAmount.toInt() + viewBinding.amount.progress
                )
            }
        })

        viewBinding.imageView.setOnTouchListener { v, event -> onTouchEvent(v, event) }
    }

    override fun changeAllInteractionState(state: Boolean) {
        viewBinding.radius.isPressed = !state
        viewBinding.amount.isPressed = !state
        viewBinding.radius.isEnabled = state
        viewBinding.amount.isEnabled = state
        viewBinding.declineBtn.isEnabled = state
        viewBinding.acceptBtn.isEnabled = state
    }

    override fun acceptButtonListener() {
        enableLoadingUi()

        uiScope.launch {
            imageRetouch()
            super.acceptButtonListener()
            disableLoadingUi()
        }
    }

    private fun onTouchEvent(view: View, event: MotionEvent): Boolean {
        val x = event.x.toInt()
        val y = event.y.toInt()

        when (event.action) {
            MotionEvent.ACTION_DOWN -> {
                pointRetouch(x, y, view)
            }
            MotionEvent.ACTION_MOVE -> {
                pointRetouch(x, y, view)
            }
        }
        return true
    }

    private suspend fun imageRetouch() {
        for (point in retouchingAlgorithm.retouchFinishPoints) {
            ImageManager.image = ImageManager.image?.let { image ->
                retouchingAlgorithm.retouching(
                    point.coordinates.x,
                    point.coordinates.y,
                    image,
                    point.radius,
                    point.sigma
                )
            }
        }
    }

    private fun pointRetouch(x: Int, y: Int, view: View): Boolean {
        val thumbnail = transformedThumbnail
        val image = ImageManager.image

        if (image != null && thumbnail != null) {
            val thumbParam = ImageParameters(thumbnail.width, thumbnail.height)
            val viewParam = ImageParameters(view.width, view.height)
            val thumbCoord = Coordinates(x, y)
            val converter = ConvertCoordinates()
            var radius = minRadius + viewBinding.radius.progress
            val sigma = minAmount + viewBinding.amount.progress

            radius = converter.convert(radius, thumbCoord, thumbParam, viewParam)

            if (converter.isNegative(thumbCoord))
                return false

            runBlocking {
                transformedThumbnail?.let {
                    updateThumbnail(
                        retouchingAlgorithm.retouching(
                            thumbCoord.x,
                            thumbCoord.y,
                            thumbnail,
                            radius,
                            sigma
                        )
                    )
                }
            }

            thumbCoord.x = (thumbCoord.x * image.width.toFloat() / thumbnail.width).toInt()
            thumbCoord.y = (thumbCoord.y * image.width.toFloat() / thumbnail.width).toInt()
            radius *= image.width.toDouble() / thumbnail.width
            retouchingAlgorithm.retouchFinishPoints.add(RetouchPointData(thumbCoord, radius, sigma))
        }

        return true
    }
}