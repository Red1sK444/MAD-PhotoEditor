package com.example.android.wonderfulapp.image.maskering

import android.os.Bundle
import android.view.View
import android.widget.SeekBar
import android.widget.SeekBar.OnSeekBarChangeListener
import by.kirich1409.viewbindingdelegate.viewBinding
import com.example.android.wonderfulapp.R
import com.example.android.wonderfulapp.databinding.FragmentMaskeringBinding
import com.example.android.wonderfulapp.image.AlgorithmFragment
import com.example.android.wonderfulapp.image.ImageManager
import com.example.android.wonderfulapp.image.helpFunc.TextViewChanger.Companion.changeParameter
import kotlinx.coroutines.launch


class MaskeringFragment : AlgorithmFragment(R.layout.fragment_maskering) {
    val minRadius = 1.0
    val minAmount = 1.0

    private val viewBinding by viewBinding(FragmentMaskeringBinding::bind, R.id.rootLayout)

    companion object {
        val TAG: String = MaskeringFragment::class.java.simpleName
        fun newInstance() = MaskeringFragment()
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        imageView = viewBinding.imageView
        loadingSpinner = viewBinding.loadingSpinner

        super.onViewCreated(view, savedInstanceState)
        super.setAcceptDeclineButtonsListeners(viewBinding.acceptBtn, viewBinding.declineBtn)

        viewBinding.radius.setOnSeekBarChangeListener(object : OnSeekBarChangeListener {
            override fun onStopTrackingTouch(seekBar: SeekBar) {}
            override fun onStartTrackingTouch(seekBar: SeekBar) {}
            override fun onProgressChanged(seekBar: SeekBar, progress: Int, fromUser: Boolean) {
                changeParameter(
                    viewBinding.radiusValue,
                    minRadius.toInt() + viewBinding.radius.progress
                )
            }
        })

        viewBinding.amount.setOnSeekBarChangeListener(object : OnSeekBarChangeListener {
            override fun onStopTrackingTouch(seekBar: SeekBar) {}
            override fun onStartTrackingTouch(seekBar: SeekBar) {}
            override fun onProgressChanged(seekBar: SeekBar, progress: Int, fromUser: Boolean) {
                changeParameter(
                    viewBinding.amountValue,
                    minAmount.toInt() + viewBinding.amount.progress
                )
            }
        })

        viewBinding.threshold.setOnSeekBarChangeListener(object : OnSeekBarChangeListener {
            override fun onStopTrackingTouch(seekBar: SeekBar) {}
            override fun onStartTrackingTouch(seekBar: SeekBar) {}
            override fun onProgressChanged(seekBar: SeekBar, progress: Int, fromUser: Boolean) {
                changeParameter(viewBinding.thresholdValue, viewBinding.threshold.progress)
            }
        })

        viewBinding.maskBtn.setOnClickListener {
            maskeringAlg()
        }
    }

    override fun acceptButtonListener() {
        enableLoadingUi()

        val radius: Double = minRadius + viewBinding.radius.progress
        val amount: Double = minAmount + viewBinding.amount.progress
        val threshold: Int = viewBinding.threshold.progress

        uiScope.launch {
            ImageManager.image =
                ImageManager.image?.let { image ->
                    MaskeringAlgorithm(image).maskering(radius, amount, threshold)
                }

            super.acceptButtonListener()
            disableLoadingUi()
        }
    }

    override fun changeAllInteractionState(state: Boolean) {
        viewBinding.radius.isPressed = !state
        viewBinding.amount.isPressed = !state
        viewBinding.threshold.isPressed = !state
        viewBinding.radius.isEnabled = state
        viewBinding.amount.isEnabled = state
        viewBinding.threshold.isEnabled = state
        viewBinding.maskBtn.isEnabled = state
        viewBinding.acceptBtn.isEnabled = state
        viewBinding.declineBtn.isEnabled = state
    }

    private fun maskeringAlg() {
        enableLoadingUi()

        uiScope.launch {
            ImageManager.thumbnail?.let {
                val ma = MaskeringAlgorithm(it)
                val radius: Double = minRadius + viewBinding.radius.progress
                val amount: Double = minAmount + viewBinding.amount.progress
                val threshold: Int = viewBinding.threshold.progress
                updateThumbnail(ma.maskering(radius, amount, threshold))
            }

            disableLoadingUi()
        }
    }
}