package com.example.android.wonderfulapp.image.colorCorrection

import android.graphics.Bitmap
import android.os.Bundle
import android.view.View
import android.widget.ImageButton
import android.widget.LinearLayout
import androidx.core.view.children
import by.kirich1409.viewbindingdelegate.viewBinding
import com.example.android.wonderfulapp.R
import com.example.android.wonderfulapp.databinding.FragmentColorCorrectionBinding
import com.example.android.wonderfulapp.image.AlgorithmFragment
import com.example.android.wonderfulapp.image.ImageManager
import kotlinx.coroutines.launch

class ColorCorrectionFragment : AlgorithmFragment(R.layout.fragment_color_correction) {

    private val viewBinding by viewBinding(FragmentColorCorrectionBinding::bind, R.id.rootLayout)
    private var currentFilter = Filters.NONE

    companion object {
        val TAG: String = ColorCorrectionFragment::class.java.simpleName
        fun newInstance() = ColorCorrectionFragment()
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        imageView = viewBinding.imageView
        loadingSpinner = viewBinding.loadingSpinner

        super.onViewCreated(view, savedInstanceState)
        super.setAcceptDeclineButtonsListeners(viewBinding.acceptBtn, viewBinding.declineBtn)

        buttonListenerBinding()
    }

    private fun buttonListenerBinding() {
        viewBinding.defaultPicture.setOnClickListener {
            currentFilter = Filters.NONE
            applyFilter(currentFilter)
        }
        viewBinding.blackWhiteFilter.setOnClickListener {
            currentFilter = Filters.BLACK_WHITE
            applyFilter(currentFilter)
        }
        viewBinding.changeContrastFilter.setOnClickListener {
            currentFilter = Filters.CHANGE_CONTRAST
            applyFilter(currentFilter)
        }
        viewBinding.negativeFilter.setOnClickListener {
            currentFilter = Filters.NEGATIVE
            applyFilter(currentFilter)
        }
        viewBinding.russiaFilter.setOnClickListener {
            currentFilter = Filters.PATRIOTIC
            applyFilter(currentFilter)
        }
        viewBinding.sepiaFilter.setOnClickListener {
            currentFilter = Filters.SEPIA
            applyFilter(currentFilter)
        }
        viewBinding.sketchFilter.setOnClickListener {
            currentFilter = Filters.SKETCH
            applyFilter(currentFilter)
        }
        viewBinding.stackBlurFilter.setOnClickListener {
            currentFilter = Filters.STACK_BLUR
            applyFilter(currentFilter)
        }
    }

    override fun initFragment() {
        super.initFragment()
        uiScope.launch {
            showExamples()
        }
    }

    private suspend fun showExamples() {
        showExample(viewBinding.defaultPicture, null)

        Filters.BLACK_WHITE.filter?.let { showExample(viewBinding.blackWhiteFilter, it) }
        Filters.NEGATIVE.filter?.let { showExample(viewBinding.negativeFilter, it) }
        Filters.PATRIOTIC.filter?.let { showExample(viewBinding.russiaFilter, it) }
        Filters.SEPIA.filter?.let { showExample(viewBinding.sepiaFilter, it) }
        Filters.SKETCH.filter?.let { showExample(viewBinding.sketchFilter, it) }
        Filters.STACK_BLUR.filter?.let { showExample(viewBinding.stackBlurFilter, it) }
        Filters.CHANGE_CONTRAST.filter?.let { showExample(viewBinding.changeContrastFilter, it) }
    }

    private suspend fun showExample(filterButton: ImageButton, filter: Filter?) {
        transformedThumbnail?.let {
            filterButton.apply {
                val bitmap = Bitmap.createScaledBitmap(
                    it,
                    maxWidth,
                    (maxWidth * (it.height.toFloat() / it.width)).toInt(),
                    true
                )
                if (filter != null) {
                    setImageBitmap(filter.apply(bitmap))
                } else {
                    setImageBitmap(bitmap)
                }
            }
        }
    }

    override fun acceptButtonListener() {
        enableLoadingUi()

        uiScope.launch {
            val image = ImageManager.image
            val filter = currentFilter.filter

            if (filter != null && image != null) {
                ImageManager.image = filter.apply(image)
            }

            super.acceptButtonListener()
            disableLoadingUi()
        }
    }

    override fun changeAllInteractionState(state: Boolean) {
        val buttons = (viewBinding.filterScrollView.getChildAt(0) as LinearLayout).children
        viewBinding.declineBtn.isEnabled = state
        viewBinding.acceptBtn.isEnabled = state
        for (btn in buttons) {
            btn.isEnabled = state
        }
    }

    private fun applyFilter(filter: Filters) {
        enableLoadingUi()

        uiScope.launch {
            when (filter) {
                Filters.NONE -> ImageManager.thumbnail?.let { updateThumbnail(it) }
                Filters.BLACK_WHITE -> blackWhiteFilter()
                Filters.CHANGE_CONTRAST -> changeContrastFilter()
                Filters.NEGATIVE -> negativeFilter()
                Filters.PATRIOTIC -> patrioticFilter()
                Filters.SEPIA -> sepiaFilter()
                Filters.SKETCH -> sketchFilter()
                Filters.STACK_BLUR -> stackBlurFilter()
            }

            disableLoadingUi()
        }
    }

    private suspend fun stackBlurFilter() {
        ImageManager.thumbnail?.let {
            val filter = StackBlurFilter()
            updateThumbnail(filter.apply(it))
        }
    }

    private suspend fun patrioticFilter() {
        ImageManager.thumbnail?.let {
            val filterRussia = PatrioticFilter()
            updateThumbnail(filterRussia.apply(it))
        }
    }

    private suspend fun blackWhiteFilter() {
        ImageManager.thumbnail?.let {
            val filter = BlackWhiteFilter()
            updateThumbnail(filter.apply(it))
        }
    }

    private suspend fun changeContrastFilter() {
        ImageManager.thumbnail?.let {
            val filter = ChangeContrastFilter()
            updateThumbnail(filter.apply(it))
        }
    }

    private suspend fun negativeFilter() {
        ImageManager.thumbnail?.let {
            val filter = NegativeFilter()
            updateThumbnail(filter.apply(it))
        }
    }

    private suspend fun sketchFilter() {
        ImageManager.thumbnail?.let {
            val filter = SketchFilter()
            updateThumbnail(filter.apply(it))
        }
    }

    private suspend fun sepiaFilter() {
        ImageManager.thumbnail?.let {
            val filter = SepiaFilter()
            updateThumbnail(filter.apply(it))
        }
    }
}