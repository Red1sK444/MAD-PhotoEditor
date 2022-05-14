package com.example.android.wonderfulapp.image

import android.os.Bundle
import android.view.View
import android.widget.ImageButton
import androidx.fragment.app.Fragment
import by.kirich1409.viewbindingdelegate.viewBinding
import com.example.android.wonderfulapp.FragmentManagerUtil
import com.example.android.wonderfulapp.R
import com.example.android.wonderfulapp.databinding.FragmentImageMainBinding
import com.example.android.wonderfulapp.image.colorCorrection.ColorCorrectionFragment
import com.example.android.wonderfulapp.image.faceDetector.FaceDetectorFragment
import com.example.android.wonderfulapp.image.filtering.FilteringFragment
import com.example.android.wonderfulapp.image.maskering.MaskeringFragment
import com.example.android.wonderfulapp.image.retouch.RetouchingFragment
import com.example.android.wonderfulapp.image.rotating.RotatingFragment
import com.example.android.wonderfulapp.image.scale.ScaleFragment


class ImageMainFragment : Fragment(R.layout.fragment_image_main) {

    private val viewBinding by viewBinding(FragmentImageMainBinding::bind, R.id.rootLayout)
    private lateinit var fragmentManagerUtil: FragmentManagerUtil

    companion object {
        val TAG: String = ImageMainFragment::class.java.simpleName
        fun newInstance() = ImageMainFragment()
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        fragmentManagerUtil = FragmentManagerUtil(parentFragmentManager, R.id.fragmentContainer)

        viewBinding.algorithmScrollView.visibility = View.INVISIBLE

        setAlgorithmNavigation()
    }

    private fun setAlgorithmNavigation() {
        setAlgorithmButtonListener(
            viewBinding.colorCorrectionBtn,
            ColorCorrectionFragment.TAG,
            ColorCorrectionFragment.newInstance()
        )
        setAlgorithmButtonListener(
            viewBinding.faceAiBtn,
            FaceDetectorFragment.TAG,
            FaceDetectorFragment.newInstance()
        )
        setAlgorithmButtonListener(
            viewBinding.filteringBtn,
            FilteringFragment.TAG,
            FilteringFragment.newInstance()
        )
        setAlgorithmButtonListener(
            viewBinding.maskeringBtn,
            MaskeringFragment.TAG,
            MaskeringFragment.newInstance()
        )
        setAlgorithmButtonListener(
            viewBinding.retouchBtn,
            RetouchingFragment.TAG,
            RetouchingFragment.newInstance()
        )
        setAlgorithmButtonListener(
            viewBinding.rotatingBtn,
            RotatingFragment.TAG,
            RotatingFragment.newInstance()
        )
        setAlgorithmButtonListener(
            viewBinding.scaleBtn,
            ScaleFragment.TAG,
            ScaleFragment.newInstance()
        )
    }

    private fun setAlgorithmButtonListener(
        button: ImageButton,
        tag: String,
        instance: AlgorithmFragment
    ) {
        button.setOnClickListener {
            fragmentManagerUtil.navigateToFragment(
                tag,
                instance
            )
        }
    }

    override fun onHiddenChanged(hidden: Boolean) {
        ImageManager.thumbnail?.let {
            viewBinding.imageView.setImageBitmap(it)
        }
    }
}