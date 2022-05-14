package com.example.android.wonderfulapp.image.rotating

import android.os.Bundle
import android.view.View
import android.widget.SeekBar
import by.kirich1409.viewbindingdelegate.viewBinding
import com.example.android.wonderfulapp.R
import com.example.android.wonderfulapp.databinding.FragmentRotatingBinding
import com.example.android.wonderfulapp.image.AlgorithmFragment
import com.example.android.wonderfulapp.image.ImageManager
import com.example.android.wonderfulapp.image.helpFunc.TextViewChanger.Companion.changeParameter
import kotlinx.coroutines.launch
import kotlin.math.IEEErem


class RotatingFragment : AlgorithmFragment(R.layout.fragment_rotating) {
    val minSeekAngle: Int = -45
    var seekAngle: Int = 0
    var curAngle: Int = 0

    private val viewBinding by viewBinding(FragmentRotatingBinding::bind, R.id.rootLayout)

    companion object {
        val TAG: String = RotatingFragment::class.java.simpleName
        fun newInstance() = RotatingFragment()
    }

    private fun refresher() {
        viewBinding.rotateSeek.progress = 45
        seekAngle = 0
        curAngle = 0
        changeParameter(
            viewBinding.rotationAngle,
            (curAngle + seekAngle).toDouble().IEEErem(360.0).toInt()
        )
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        imageView = viewBinding.imageView
        loadingSpinner = viewBinding.loadingSpinner

        super.onViewCreated(view, savedInstanceState)
        super.setAcceptDeclineButtonsListeners(viewBinding.acceptBtn, viewBinding.declineBtn)

        viewBinding.leftBtn.setOnClickListener {
            curAngle = (curAngle + 90) % 360
            rotate(curAngle + seekAngle)
            changeParameter(
                viewBinding.rotationAngle,
                (curAngle + seekAngle).toDouble().IEEErem(360.0).toInt()
            )
        }

        viewBinding.rightBtn.setOnClickListener {
            curAngle = (curAngle - 90) % 360
            rotate(curAngle + seekAngle)
            changeParameter(
                viewBinding.rotationAngle,
                (curAngle + seekAngle).toDouble().IEEErem(360.0).toInt()
            )
        }

        viewBinding.rotateSeek.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onStopTrackingTouch(seekBar: SeekBar) {}
            override fun onStartTrackingTouch(seekBar: SeekBar) {}
            override fun onProgressChanged(seekBar: SeekBar, progress: Int, fromUser: Boolean) {
                if (fromUser) {
                    seekAngle = minSeekAngle + viewBinding.rotateSeek.progress
                    changeParameter(
                        viewBinding.rotationAngle,
                        (curAngle + seekAngle).toDouble().IEEErem(360.0).toInt()
                    )
                    rotate(curAngle + seekAngle)
                }
            }
        })
    }

    override fun acceptButtonListener() {
        enableLoadingUi()

        uiScope.launch {
            ImageManager.image =
                ImageManager.image?.let { image ->
                    RotatingAlgorithm(image).rotate(curAngle + seekAngle)
                }

            super.acceptButtonListener()
            disableLoadingUi()
        }
    }

    override fun changeAllInteractionState(state: Boolean) {
        viewBinding.rotateSeek.isPressed = !state
        viewBinding.rotateSeek.isEnabled = state
        viewBinding.leftBtn.isEnabled = state
        viewBinding.rightBtn.isEnabled = state
        viewBinding.declineBtn.isEnabled = state
        viewBinding.acceptBtn.isEnabled = state
    }

    private fun rotate(angle: Int) {
        enableLoadingUi()

        uiScope.launch {
            ImageManager.thumbnail?.let {
                val rotate = RotatingAlgorithm(it)
                updateThumbnail(rotate.rotate(angle))
            }
            disableLoadingUi()
        }
    }
}