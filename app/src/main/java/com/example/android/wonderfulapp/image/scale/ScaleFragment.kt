package com.example.android.wonderfulapp.image.scale

import android.os.Bundle
import android.view.View
import android.widget.SeekBar
import androidx.core.content.res.ResourcesCompat
import by.kirich1409.viewbindingdelegate.viewBinding
import com.example.android.wonderfulapp.R
import com.example.android.wonderfulapp.databinding.FragmentScaleBinding
import com.example.android.wonderfulapp.image.AlgorithmFragment
import com.example.android.wonderfulapp.image.ImageManager
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import kotlin.math.sqrt

// Как же тут все плохо, помогите
class ScaleFragment : AlgorithmFragment(R.layout.fragment_scale), SeekBar.OnSeekBarChangeListener {
    private val viewBinding by viewBinding(FragmentScaleBinding::bind, R.id.rootLayout)

    private lateinit var scaleAlgorithm: ScaleAlgorithm
    private var curMinProgress = 0
    private var curFactor: Float = 1F

    companion object {
        val TAG: String = ScaleFragment::class.java.simpleName
        fun newInstance() = ScaleFragment()
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        imageView = viewBinding.imageView
        loadingSpinner = viewBinding.loadingSpinner

        super.onViewCreated(view, savedInstanceState)
        super.setAcceptDeclineButtonsListeners(viewBinding.acceptBtn, viewBinding.declineBtn)

        viewBinding.factorSeekBar.setOnSeekBarChangeListener(this)
    }

    override fun initFragment() {
        super.initFragment()

        curFactor = 1F
        curMinProgress = calculateMinProgress()
        viewBinding.factorSeekBar.max = calculateMaxProgress()
        viewBinding.factorSeekBar.progress = 9 - curMinProgress

        updateImageRatioTextView()
        ImageManager.thumbnail?.let {
            scaleAlgorithm = ScaleAlgorithm(it)
            scaleAlgorithm.initMipmap()
        }
    }

    override fun acceptButtonListener() {
        enableLoadingUi()

        uiScope.launch {
            ImageManager.image =
                ImageManager.image?.let { image ->
                    ScaleAlgorithm(image).scaleImage(curFactor)
                }

            super.acceptButtonListener()
            ImageManager.reDownscaleThumbnail()

            disableLoadingUi()
        }
    }

    override fun onStartTrackingTouch(seekBar: SeekBar?) {}

    override fun onStopTrackingTouch(seekBar: SeekBar?) {}

    override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
        curFactor = calculateFactor(viewBinding.factorSeekBar.progress)

        updateFactorTextView()
        updateImageRatioTextView()

        enableLoadingUi()

        uiScope.launch {
            updateThumbnail(scaleAlgorithm.scaleImage(curFactor))
            disableLoadingUi()
        }
    }

    override fun changeAllInteractionState(state: Boolean) {
        viewBinding.factorSeekBar.isPressed = !state
        viewBinding.factorSeekBar.isEnabled = state
        viewBinding.acceptBtn.isEnabled = state
        viewBinding.declineBtn.isEnabled = state
    }

    private fun calculateMaxProgress(): Int {
        var maxFactor = 1F

        ImageManager.image?.let {
            val realMaxFactor = sqrt(ImageManager.imageMaxSize / (it.width * it.height).toFloat())
            maxFactor = (realMaxFactor - realMaxFactor.rem(1.5F)).coerceAtLeast(1F)
        }

        return ((maxFactor / 0.5).toInt() + 7) - curMinProgress
    }

    private fun calculateMinProgress(): Int {
        var minFactor = 1F

        ImageManager.image?.let {
            val realMinFactor = sqrt(ImageManager.imageMinSize / (it.width * it.height).toFloat())
            minFactor = realMinFactor - realMinFactor.rem(0.1F)
        }

        return (minFactor * 10).toInt()
    }

    private fun updateImageRatioTextView() {
        ImageManager.image?.let {
            viewBinding.imageRatio.text = getString(
                R.string.scale_image_ratio,
                (it.width * curFactor).toInt(),
                (it.height * curFactor).toInt()
            )
        }
    }

    private fun updateFactorTextView() {
        viewBinding.factorValue.text = curFactor.toString()
    }

    private fun calculateFactor(progress: Int): Float {
        val realProgress = curMinProgress + progress

        return if (realProgress < 10) {
            (realProgress + 1) / 10F
        } else {
            (realProgress - 10) * 0.5F + 1.5F
        }
    }
}