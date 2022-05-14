package com.example.android.wonderfulapp

import android.app.Activity
import android.content.Context
import android.view.View
import android.view.inputmethod.InputMethodManager

object UiUtils {
    fun hideKeyboard(context: Activity) {
        val view: View = context.findViewById(android.R.id.content)
        val imm: InputMethodManager =
            context.getSystemService(Context.INPUT_METHOD_SERVICE) as InputMethodManager
        imm.hideSoftInputFromWindow(view.windowToken, 0)
    }
}