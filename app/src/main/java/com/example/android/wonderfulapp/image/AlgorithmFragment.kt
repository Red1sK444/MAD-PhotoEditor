package com.example.android.wonderfulapp.image

import android.graphics.Bitmap
import android.os.Bundle
import android.view.View
import android.widget.ImageButton
import android.widget.ImageView
import androidx.fragment.app.Fragment
import com.example.android.wonderfulapp.FragmentManagerUtil
import com.example.android.wonderfulapp.R
import com.github.ybq.android.spinkit.SpinKitView
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job

abstract class AlgorithmFragment(layoutId: Int) : Fragment(layoutId) {

    protected var transformedThumbnail: Bitmap? = ImageManager.thumbnail
    protected var imageView: ImageView? = null
    protected var loadingSpinner: SpinKitView? = null

    private val updateImageJob = Job()
    protected val uiScope = CoroutineScope(Dispatchers.Main + updateImageJob)

    private lateinit var fragmentManagerUtil: FragmentManagerUtil

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        fragmentManagerUtil = FragmentManagerUtil(parentFragmentManager, R.id.fragmentContainer)
        initFragment()
    }

    override fun onHiddenChanged(hidden: Boolean) {
        if (!hidden) {
            initFragment()
        }
    }

    open fun initFragment() {
        transformedThumbnail = ImageManager.thumbnail
        imageView?.setImageBitmap(transformedThumbnail)
    }

    open fun updateThumbnail(image: Bitmap) {
        transformedThumbnail = image
        imageView?.setImageBitmap(transformedThumbnail)
    }

    fun setAcceptDeclineButtonsListeners(acceptBtn: ImageButton, declineBtn: ImageButton) {
        acceptBtn.setOnClickListener {
            acceptButtonListener()
        }

        declineBtn.setOnClickListener {
            declineButtonListener()
        }
    }

    open fun acceptButtonListener() {
        ImageManager.thumbnail = transformedThumbnail
        navigateBack()
    }

    open fun declineButtonListener() {
        navigateBack()
    }

    private fun navigateBack() {
        fragmentManagerUtil.navigateToFragment(
            ImageMainFragment.TAG,
            ImageMainFragment.newInstance()
        )
    }

    open fun enableLoadingUi() {
        changeAllInteractionState(false)
        showLoadingOnImage()
    }

    open fun disableLoadingUi() {
        hideLoadingOnImage()
        changeAllInteractionState(true)
    }

    abstract fun changeAllInteractionState(state: Boolean)

    private fun hideLoadingOnImage() {
        loadingSpinner?.visibility = View.GONE
        imageView?.alpha = 1F
        imageView?.clearColorFilter()
    }

    private fun showLoadingOnImage() {
        loadingSpinner?.visibility = View.VISIBLE
        imageView?.alpha = 0.9F
        imageView?.setColorFilter(
            resources.getColor(R.color.grey_500, null),
            android.graphics.PorterDuff.Mode.MULTIPLY
        )
    }
}