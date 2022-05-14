package com.example.android.wonderfulapp.figure

import android.graphics.PixelFormat
import android.os.Bundle
import android.view.View
import android.view.View.GONE
import android.view.View.VISIBLE
import android.widget.SeekBar
import androidx.fragment.app.Fragment
import by.kirich1409.viewbindingdelegate.viewBinding
import com.example.android.wonderfulapp.R
import com.example.android.wonderfulapp.databinding.FragmentFigureBinding
import com.example.android.wonderfulapp.figure.model3D.Figure
import com.example.android.wonderfulapp.figure.polyhedra.*

class FigureFragment : Fragment(R.layout.fragment_figure) {

    companion object {
        val TAG: String = FigureFragment::class.java.simpleName
        fun newInstance() = FigureFragment()
    }

    private val viewBinding by viewBinding(FragmentFigureBinding::bind, R.id.rootLayout)
    private val polyhedrons: MutableList<Figure?> = MutableList(21) { null }

    init {
        polyhedrons[4] = Tetrahedron
        polyhedrons[6] = Cube
        polyhedrons[8] = Octahedron
        polyhedrons[20] = Icosahedron
        polyhedrons[12] = Dodecahedron
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        viewBinding.figureSurfaceView.holder.addCallback(viewBinding.figureSurfaceView)

        viewBinding.faceCountBar.setOnSeekBarChangeListener(object :
            SeekBar.OnSeekBarChangeListener {

            override fun onStartTrackingTouch(seekBar: SeekBar?) {
                viewBinding.facesValue.visibility = VISIBLE
                touchOnFaceCountBar(seekBar)
            }

            override fun onStopTrackingTouch(seekBar: SeekBar?) {
                viewBinding.facesValue.visibility = GONE
            }

            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                moveOnFaceCountBar(seekBar, progress)
            }
        })

        viewBinding.rotationZBar.setOnSeekBarChangeListener(object :
            SeekBar.OnSeekBarChangeListener {

            override fun onStartTrackingTouch(seekBar: SeekBar?) {}
            override fun onStopTrackingTouch(seekBar: SeekBar?) {}

            var prevVal = 0
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                if (!fromUser) {
                    prevVal = progress - 180
                    return
                }

                val currentVal = progress - 180
                viewBinding.figureSurfaceView.rotateByZ(currentVal - prevVal)
                prevVal = currentVal
            }
        })
    }

    private fun touchOnFaceCountBar(seekBar: SeekBar?) {
        seekBar?.let {
            if (viewBinding.facesValue.text == "") {
                viewBinding.facesValue.text = (it.progress + 4).toString()
                moveSeekbarHint(it)
            }
        }
    }

    private fun moveOnFaceCountBar(seekBar: SeekBar?, progress: Int) {
        val faceCount = 4 + progress
        viewBinding.facesValue.text = faceCount.toString()

        seekBar?.let {
            moveSeekbarHint(seekBar)
        }

        if (polyhedrons[faceCount] == null) {
            polyhedrons[faceCount] = CustomFigure(faceCount)
        }

        polyhedrons[faceCount]?.let {
            viewBinding.figureSurfaceView.changeFigure(it)
            viewBinding.rotationZBar.progress = it.rotationZ + 180
        }
    }

    private fun moveSeekbarHint(seekBar: SeekBar) {
        val offset =
            (seekBar.width - seekBar.paddingLeft - seekBar.paddingRight) * seekBar.progress / seekBar.max
        viewBinding.facesValue.x =
            seekBar.x + seekBar.paddingLeft + offset - viewBinding.facesValue.width / 2
    }

    override fun onHiddenChanged(hidden: Boolean) {
        viewBinding.figureSurfaceView.holder.setFormat(if (!hidden) PixelFormat.OPAQUE else PixelFormat.TRANSPARENT)
    }
}