package com.example.android.wonderfulapp.image.helpFunc

import android.widget.TextView

class TextViewChanger {
    companion object{
        fun changeParameter(tv: TextView, value:Int){
            tv.text = value.toString()
        }
    }
}