package com.example.android.wonderfulapp.spline

import android.os.Bundle
import android.view.View
import androidx.fragment.app.Fragment
import by.kirich1409.viewbindingdelegate.viewBinding
import com.example.android.wonderfulapp.R
import com.example.android.wonderfulapp.databinding.FragmentSplineBinding

class SplineFragment : Fragment(R.layout.fragment_spline) {

    companion object {
        val TAG: String = SplineFragment::class.java.simpleName
        fun newInstance() = SplineFragment()
    }

    private val viewBinding by viewBinding(FragmentSplineBinding::bind, R.id.rootLayout)

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {

        super.onViewCreated(view, savedInstanceState)

        viewBinding.clearField.setOnClickListener { viewBinding.splineView.clearCanvas() }
        viewBinding.addPoint.setOnClickListener { viewBinding.splineView.setAddPointState() }
        viewBinding.movePoint.setOnClickListener { viewBinding.splineView.setMovePointState() }
        viewBinding.deletePoint.setOnClickListener { viewBinding.splineView.setDeletePointState() }
        viewBinding.drawSpline.setOnClickListener { viewBinding.splineView.transformToSpline() }
    }

    override fun onHiddenChanged(hidden: Boolean) {
        viewBinding.splineView.visibility = if (!hidden) View.VISIBLE else View.INVISIBLE
    }
}
